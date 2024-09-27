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

// Device function to perform 8-point FFT using warp shuffle
__device__ void synch_fft8_new(cuFloatComplex *data) {
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
    unsigned int j = bitReverse(tidX, numBits);

    
    // Use shuffle to get the value of thread j
    cuFloatComplex xj;
    xj.x = __shfl_sync(0xFF, x.x,tidBlockY + j);
    xj.y = __shfl_sync(0xFF, x.y, tidBlockY + j);


    /*
    // Write the potentially swapped value back to data
    data[tidX] = x;

    // Synchronize to ensure all swaps are done
    __syncwarp();
    */

    /*
    -----------------------------------------
        Step 2: Iterative FFT Stages
    -----------------------------------------
    */ 
    // CHECK FROM HERE
    for(unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s;          // m = 2,4,8
        unsigned int m2 = m >> 1;          // m/2 = 1,2,4
        float angle = -2.0f * M_PIf / (float)m; // Twiddle factor angle
        cuFloatComplex wm = make_cuFloatComplex(cosf(angle), sinf(angle)); // wm = e^{-2πi/m}

        // Determine the group and position within the group
        unsigned int group = tidX / m2;
        unsigned int pos = tidX % m2;

        // Compute twiddle factor w = wm^pos
        // Efficient computation using angle multiplication
        float angle_pos = pos * angle;
        cuFloatComplex w = make_cuFloatComplex(cosf(angle_pos), sinf(angle_pos));

        // Determine partner index for butterfly
        unsigned int partner = tidX ^ m2;

        // Read partner's value using shuffle
        cuFloatComplex xj;
        xj.x = __shfl_sync(0xFF, x.x, partner);
        xj.y = __shfl_sync(0xFF, x.y, partner);

        // Compute t = w * xj
        cuFloatComplex t = cuCmulf(w, xj);

        // Compute u = x (current thread's value)
        cuFloatComplex u = x;

        // Compute the new values for x[i] and x[j]
        cuFloatComplex new_xi = cuCaddf(u, t); // x[i] = u + t
        cuFloatComplex new_xj = cuCsubf(u, t); // x[j] = u - t

        // Update x based on thread's position in the butterfly
        if(tidX < partner) {
            x = new_xi;
        } else {
            x = new_xj;
        }

        // Write the updated value back to data
        data[tidX] = x;

        // Synchronize threads before the next stage
        __syncwarp();
    }
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
    // Use shuffle to get the value of thread j
    /*
        cuFloatComplex xj;
        xj.x = __shfl_sync(0xFF, x.x,tidBlockY + j);
        xj.y = __shfl_sync(0xFF, x.y, tidBlockY + j);
    */

    /*
    // Write the potentially swapped value back to data
    data[tidX] = x;

    // Synchronize to ensure all swaps are done
    __syncwarp();
    */

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
            printf("Thread tidx= %d; tidBlocky = %d; s = %d; u = (%.2f,%.2f); t = (%.2f,%.2f)\n", tidX, tidBlockY, s, u.x, u.y, t.x, t.y);
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

    int globalX = blockIdx.x * BLOCK_SIZE + localX;
    int globalY = blockIdx.y * BLOCK_SIZE + localY;
    int globalIdx = globalY * width + globalX;

    if (globalX < width && globalY < height) {
        blockData[localY][localX] = input[globalIdx];
    } else {
        // Pad with zeros but should never happen due to initial checks
        blockData[localY][localX] = make_cuFloatComplex(0, 0);
    }

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
    int localIdx = localY * BLOCK_SIZE + localX;
    blockData[localY][localX] = cuCdivf(blockData[localY][localX], quantizationMatrix[localIdx]);
    
    // __syncthreads();

    // Write back to global memory
    if (globalX < width && globalY < height) {
        output[globalIdx] = blockData[localY][localX];
    }
}