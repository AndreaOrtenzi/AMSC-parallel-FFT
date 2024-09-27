#include <cstddef>
#include <cstdio>
#include "utils.cu"


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
    constexpr unsigned int numBits = LOG_BLOCK_SIZE;

    // Each thread index
    const unsigned int tidBlockY = threadIdx.y * blockDim.x;
    const unsigned int tidX = threadIdx.x;

    // Load data from global memory into registers
    cuFloatComplex x = data[tidX];

    /* 
    -----------------------------------------
        Step 1: Bit-Reversal Permutation
    -----------------------------------------
    */
    unsigned int j = bitReverse(tidX, numBits); // Checked
    __syncwarp();

    // Use shuffle instead of writing back to data (data[j] = x)
    /*
        In shuffle operations each thread of the warp is represented as a lane.
        t0  ------------------------------
        t1  ------------------------------
        t2  ------------------------------
        ...
        t31 ------------------------------
        In this case j is from 0 to 7 because the 32 threads are divided into 4 rows of 8 elements.
        To correctly exchange the values of x and xj, we need to add 8*32//warpId to j.
        Notice that
        - 0-31 numbers in binary are 00000-11111
            Warp id can then be calculated taking the first 5 bits of the localThreadId
        - 0-7 numbers in binary are 000-111
            From this we can see that the first 2 bits identify the row and the last 3 bits identify the column
        We can then calculate the correct j by adding the 4th and 5th bits of the localThreadId to the j.
        Apply a mask 00..0011000 = 0x18 HEX = 24 DEC
    */
    const unsigned int maskedLocalIdx = (tidBlockY + tidX) & 0x18;
    j += maskedLocalIdx;

    // Use shuffle to get the value of thread j
    x.x = __shfl_sync(0xFF, x.x, j);
    x.y = __shfl_sync(0xFF, x.y, j);

    /*
    -----------------------------------------
        Step 2: Iterative FFT Stages
    -----------------------------------------
    */
    size_t maskRootsIdx = 0;
    cuFloatComplex &localData = x;
    for (unsigned s = 0; s < numBits; s++) { // Three stages for FFT of size 8

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
        // cuFloatComplex wm = rootsOfUnity[(threadIdx.x % halfStep)*(BLOCK_SIZE/step)];// 0-7 % (1,2,4,8) * (8/2 = 4, 2=8/4, 1=8/8)
        
        // 000 001 010 011 100 101 110 111 -> 000
        // 000 001 010 011 100 101 110 111 %2 -> look at the last bit
        // 000 001 000 001 000 001 000 001 %4 -> look at the last 2 bits
        // 8 >> 1 = 4, 8 >> 2 = 2, 8 >> 3 = 1
        // *4 = << 2, *2 = << 1, *1 = << 0
        // numBits - s - 1 = 2, 1, 0
        
        size_t rootsOfUnityIdx = threadIdx.x & maskRootsIdx; // threadIdx.x % halfStep
        rootsOfUnityIdx <<= numBits - s - 1; // *(BLOCK_SIZE/step)

        cuFloatComplex wm = rootsOfUnity[rootsOfUnityIdx];
        maskRootsIdx += 1<<s;

        cuFloatComplex pairedData;
        pairedData.x = __shfl_sync(0xFF, x.x, pairedIdx + maskedLocalIdx);
        pairedData.y = __shfl_sync(0xFF, x.y, pairedIdx + maskedLocalIdx);

        // t if hasElemU, -t if hasElemT
        cuFloatComplex t = cuCsubf(cuCmulf(cuCmulf(wm, pairedData), hasElemU), cuCmulf(hasElemT, cuCmulf(wm, localData)));
        // u if hasElemT, u if hasElemU
        cuFloatComplex u = cuCaddf(cuCmulf(localData, hasElemU), cuCmulf(pairedData, hasElemT));
        
        localData = cuCaddf(t, u);

        #if DEBUG
            if (tidBlockY == 0) {
                printf("Thread tidx= %d; tidBlocky = %d; s = %d; u = (%.2f,%.2f); t = (%.2f,%.2f)\n", tidX, tidBlockY, s, u.x, u.y, t.x, t.y);
            }
        #endif
    }
    data[threadIdx.x] = localData;

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