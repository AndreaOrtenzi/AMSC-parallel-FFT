#include <cassert>
#include <cuda_runtime.h>
#include <cuComplex.h>

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 8  // 8x8 tiles
#define LOG_BLOCK_SIZE 3
#endif

#ifndef NUM_TILE_X_THREAD_BLOCK // Available from version 4
#define NUM_TILE_X_THREAD_BLOCK 8
#endif

#ifndef DEBUG
#define DEBUG true
#endif

// Kernel to reorganize the matrix so that 8x8 submatrices are stored contiguously in memory
__global__ void reorganizeSubblocks(const cuFloatComplex* input, cuFloatComplex* output, int width, int height, bool reorganizeToContiguous) {
    assert(blockDim.x == BLOCK_SIZE && blockDim.y == BLOCK_SIZE && "Block dim is different form the defined one");
    // Compute the global index for the submatrix (tile)
    int tileX = blockIdx.x;
    int tileY = blockIdx.y;

    // Compute the global indices for the element within the submatrix
    int localX = threadIdx.x;
    int localY = threadIdx.y;

    // Global position of the element in the input matrix
    int globalX = tileX * BLOCK_SIZE + localX;
    int globalY = tileY * BLOCK_SIZE + localY;

    int numBlocksX = width / BLOCK_SIZE;

    // Ensure we are within bounds
    if (globalX < width && globalY < height) {
        // Compute the index in the original input matrix (row-major order)
        int bigMatrixIndex = globalY * width + globalX;

        // Compute the new index in the output matrix where submatrices are contiguous
        int tileIndex = tileY * numBlocksX + tileX;  // Which tile we're in
        int contiguousIndex = tileIndex * BLOCK_SIZE * BLOCK_SIZE + localY * BLOCK_SIZE + localX;

        // Reorganize the data
        if (reorganizeToContiguous)
            output[contiguousIndex] = input[bigMatrixIndex];
        else
            output[bigMatrixIndex] = input[contiguousIndex];
    }
}

// Error checking
#define CUDA_CHECK_ERROR(call)                                              \
    {                                                                       \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in function " << __PRETTY_FUNCTION__   \
                      << ", file " << __FILE__ << "-" << __LINE__ << ":\n\t"\
                      << cudaGetErrorString(err) << std::endl;              \
            exit(err);                                                      \
        }                                                                   \
    }

// Constants for quantization (example values)
__constant__ float2 quantizationMatrix[64] = {
    // 8x8 quantization matrix values
    {16},{11},{10},{16},{24},{40},{51},{61},
    {12},{12},{14},{19},{26},{58},{60},{55},
    {14},{13},{16},{24},{40},{57},{69},{56},
    {14},{17},{22},{29},{51},{87},{80},{62},
    {18},{22},{37},{56},{68},{109},{103},{77},
    {24},{35},{55},{64},{81},{104},{113},{92},
    {49},{64},{78},{87},{103},{121},{120},{101},
    {72},{92},{95},{98},{112},{100},{103},{99}
};

// Roots of unity for 8-point FFT
// Last element is a duplicate of the first element to allow inverse FFT
__constant__ float2 rootsOfUnity[9] = {
    {1, 0},
    {0.707107f, -0.707107f},
    {0, -1},
    {-0.707107f, -0.707107f},
    {-1, 0},
    {-0.707107f, 0.707107f},
    {0, 1},
    {0.707107f, 0.707107f},
    {1.f, 0.f}
};

// Kernel to quantize cufft results
__global__ void quantizeKernel(cuFloatComplex *input, cuFloatComplex *output, int width, int height) {
    // Calculate the global indices for the 8x8 subblocks

    // Local thread indices
    // 8x8
    const int localIdx = threadIdx.y * blockDim.x + threadIdx.x;

    // Version with contiguous 8x8 submatrices
    const int numTilesX = width / BLOCK_SIZE;   // Number of horizontal 8x8 tiles in the matrix
    const int tileIdx = blockIdx.y * numTilesX + blockIdx.x;  // Flat index of the current tile

    // Compute the offset for the current element within the submatrix
    const int tileOffset = tileIdx * BLOCK_SIZE * BLOCK_SIZE;

    // Compute the global index in the reorganized layout
    int globalIdx = tileOffset + localIdx;

    // Apply quantization
    output[globalIdx] = cuCdivf(input[globalIdx], quantizationMatrix[localIdx]);
}

#define TILE_DIM 8
__global__ void transpose(cuFloatComplex *odata, const cuFloatComplex *idata)
{
  __shared__ cuFloatComplex tile[TILE_DIM * TILE_DIM];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += blockDim.y)
     tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  for (int j = 0; j < TILE_DIM; j += blockDim.y)
     odata[(y+j)*width + x] = tile[(threadIdx.y+j)*TILE_DIM + threadIdx.x];          
}