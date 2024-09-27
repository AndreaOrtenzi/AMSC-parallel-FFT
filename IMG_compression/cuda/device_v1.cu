#include <cstdio>
#include "utils.cu"

/*
// Device function for 8-point FFT
__device__ void fft8(cuFloatComplex *data) {
    // Implement Cooley-Tukey FFT algorithm for 8 elements
    constexpr int n = BLOCK_SIZE;
    for (int s = 0; s < 3; s++) {
        // int m = 1 << s;
        int m2 = 1 << s;
        cuFloatComplex wm = rootsOfUnity[m2];
        for (int k = threadIdx.x; k < n; k += BLOCK_SIZE) {
            for (int j = 0; j < m2; j++) {
                cuFloatComplex t = cuCmulf(wm, data[k + j + m2]);
                cuFloatComplex u = data[k + j];
                data[k + j] = cuCaddf(u, t);
                data[k + j + m2] = cuCsubf(u, t);
            }
        }
        __syncthreads();
    }
}*/

/*
    Helper function to perform bit reversal
    Example: n = 8, numBits = 3
    x = [0, 1, 2, 3, 4, 5, 6, 7]
    out = [0, 4, 2, 6, 1, 5, 3, 7]
*/
__device__ __forceinline__ unsigned int bitReverse(unsigned int x, const unsigned int log2n) {
    unsigned int j = 0;
    for(unsigned int k = 0; k < log2n; k++) {
        j = (j << 1) | ((x  >> k) & 1U);
    }
    return j;
}

__device__ void synch_fft8(cuFloatComplex *data) {
    // Number of points
    constexpr unsigned int n = BLOCK_SIZE;
    constexpr unsigned int numBits = LOG_BLOCK_SIZE;

    // Each thread index
    unsigned int tidBlockY = threadIdx.y * blockDim.x;
    unsigned int tidX = threadIdx.x;

    // Load data from global memory into registers
    cuFloatComplex x = data[tidX];

    /* 
    -----------------------------------------
        Step 1: Bit-Reversal Permutation
    -----------------------------------------
    */
    unsigned int j = bitReverse(tidX, numBits); // Checked
    __syncwarp();

    data[j] = x;
    __syncwarp();

    /*
    -----------------------------------------
        Step 2: Iterative FFT Stages
    -----------------------------------------
    */ 
    
    for (unsigned s = 0; s < numBits; s++) { // Three stages for FFT of size 8
        unsigned step = 1 << (s+1); // Step size for each butterfly group
        unsigned halfStep = 1 << (s);

        // Synchronize threads before starting the stage
        __syncwarp();
        
        // Toggle the s-1 bit for the butterfly group
        unsigned pairedIdx = threadIdx.x ^ (1U << s);
        
        // This value is 1 if it has element u in the butterfly group, 0 otherwise
        cuFloatComplex hasElemU = make_cuFloatComplex(static_cast<float>(threadIdx.x < pairedIdx),0.);
        // This value is 1 if it has element t in the butterfly group, 0 otherwise
        cuFloatComplex hasElemT = make_cuFloatComplex(static_cast<float>(threadIdx.x > pairedIdx),0.);

        // 8 - to reverse the order of the roots of unity
        // First root is duplicated in the 8th index
        // cuFloatComplex wm = rootsOfUnity[8 - (threadIdx.x % halfStep)*(8/step)];
        cuFloatComplex wm = rootsOfUnity[(threadIdx.x % halfStep)*(8/step)];

        cuFloatComplex localData = data[threadIdx.x];
        cuFloatComplex pairedData = data[pairedIdx];

        // t if hasElemU, -t if hasElemT
        cuFloatComplex t = cuCsubf(cuCmulf(cuCmulf(wm, pairedData), hasElemU), cuCmulf(hasElemT, cuCmulf(wm, localData)));
        // u if hasElemT, u if hasElemU
        cuFloatComplex u = cuCaddf(cuCmulf(localData, hasElemU), cuCmulf(pairedData, hasElemT));
        
        data[threadIdx.x] = cuCaddf(t, u);

        #if DEBUG
            if (tidBlockY == 0) {
                printf("Thread tidx= %d; tidBlocky = %d; s = %d; u = (%.2f,%.2f); t = (%.2f,%.2f)\n", tidX, tidBlockY, s, u.x, u.y, t.x, t.y);
            }
        #endif
        
        /*
        // Each thread handles multiple butterfly groups
        for (int k = threadIdx.x * step; k < n; k += blockDim.x * step) {
            for (int j = 0; j < halfStep; j++) {
                int index = k + j;
                int pair = index + halfStep;

                // Ensure indices are within bounds
                if (pair < n) {
                    // Compute twiddle factor
                    cuFloatComplex wm = rootsOfUnity[j * (8 / step)];
                    
                    // Perform butterfly operations
                    cuFloatComplex t = cuCmulf(wm, data[pair]);
                    cuFloatComplex u = data[index];
                    
                    data[index] = cuCaddf(u, t);
                    data[pair] = cuCsubf(u, t);
                }
            }
        }*/
    }

    // Final synchronization after all stages
    __syncthreads();
}


__global__ void fftQuantizeKernel(cuFloatComplex *input, cuFloatComplex *output, int width, int height) {

    // Shared memory for an 8x8 subblock
    __shared__ cuFloatComplex blockData[BLOCK_SIZE][BLOCK_SIZE];

    // Load data into shared memory
    int localX = threadIdx.x;
    int localY = threadIdx.y;
    int localIdx = localY * BLOCK_SIZE + localX;

    int globalX = blockIdx.x * BLOCK_SIZE + localX;
    int globalY = blockIdx.y * BLOCK_SIZE + localY;

    // Version with original matrix
    /* 
    int globalIdx = globalY * width + globalX;
    */

    // Version with contiguous 8x8 submatrices
    int numTilesX = width / BLOCK_SIZE;   // Number of horizontal 8x8 tiles in the matrix
    int tileIdx = blockIdx.y * numTilesX + blockIdx.x;  // Flat index of the current tile

    // Compute the offset for the current element within the submatrix
    int tileOffset = tileIdx * BLOCK_SIZE * BLOCK_SIZE;

    // Compute the global index in the reorganized layout
    int globalIdx = tileOffset + localIdx;


    assert(globalX < width && globalY < height && "Input matrix should be divisible by 8");
    blockData[localY][localX] = input[globalIdx];

    __syncthreads();

    // Perform FFT on rows
    synch_fft8(&blockData[localY][0]);

    #if DEBUG
    if (globalIdx==0) {
        for (int row = 0; row < BLOCK_SIZE; ++row) {
            for (int col = 0; col < BLOCK_SIZE; ++col) {
                printf("(%.2f,%.2f) ", blockData[row][col].x, blockData[row][col].y);
            }
            printf("\n");
        }
    }
    __syncthreads();
    #endif

    // Transpose the block for column-wise FFT
    cuFloatComplex temp = blockData[localX][localY];
    __syncthreads();
    blockData[localY][localX] = temp;
    __syncthreads();

    // Perform FFT on columns
    synch_fft8(&blockData[localY][0]);

    // Transpose back    
    temp = blockData[localX][localY];
    __syncthreads();
    blockData[localY][localX] = temp;
    __syncthreads();

    
    // Apply quantization
    blockData[localY][localX] = cuCdivf(blockData[localY][localX], quantizationMatrix[localIdx]);

    // Write back to global memory
    if (globalX < width && globalY < height) {
        output[globalIdx] = blockData[localY][localX];
    }
}