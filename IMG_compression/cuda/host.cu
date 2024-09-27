#include <cstddef>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <memory>
#include <random>
#include <chrono>

#ifndef USE_CUFFT
#define USE_CUFFT true
#endif

#ifndef VERSION
#define VERSION 1
#endif

#ifndef iTT
#define iTT 1
#endif

#if USE_CUFFT
#include <cufft.h>
#endif

#if VERSION == 2
#include "device_v2.cu"
#elif VERSION == 3
#include "device_v3.cu"
#elif VERSION == 4
#include "device_v4.cu"
#else
#include "device_v1.cu"
#endif

/*
Generate data for testing
Use DATA to switch between random and fixed data generation
Cases:
    DATA = 0: All 0s
    DATA = 1: Same element value for all elements
    DATA = 123: Random data generation
*/
template <unsigned DATA>
void generateData(cuFloatComplex* data, int width, int height) {
    size_t matrixSize = width * height;
    if constexpr (DATA == 0) {
        for (int i = 0; i < matrixSize; ++i) {
            data[i] = make_cuFloatComplex(0., 0.);;
        }
    } else if constexpr (DATA == 1) {
        for (int i = 0; i < matrixSize; ++i) {
            data[i] = make_cuFloatComplex(3., 0.);;
        }
    } else if constexpr (DATA == 123) {
        std::mt19937 gen(20); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dis(0, 255);

        for (size_t i = 0; i < matrixSize; ++i) {
            float val = static_cast<float>(dis(gen));
            data[i] = make_cuFloatComplex(val, 0);
        }
    } else if (DATA == 2) {
        
        constexpr unsigned MAT[8][8] = {
            140, 151, 183, 216, 154, 219, 139, 216,
            108, 159, 165, 98, 112, 76, 228, 14,
            246, 69, 98, 122, 202, 207, 135, 122,
            145, 100, 236, 214, 18, 86, 22, 165,
            5, 94, 213, 245, 199, 35, 222, 222,
            250, 121, 204, 205, 118, 133, 199, 173,
            30, 184, 163, 148, 36, 137, 241, 194,
            133, 27, 106, 121, 67, 47, 198, 188
        };
        for (int i = 0; i < height; ++i) {
            int mati = i % BLOCK_SIZE;
            for (int j = 0; j < width; ++j) {
                int matj = j % BLOCK_SIZE;
                data[i*width + j] = make_cuFloatComplex(MAT[mati][matj], 0.);
            }
        }

    }
}

void displayMatrix10x10(cuFloatComplex* mat, int width, int height) {
    std::cout << "Top-left 10x10 corner (real part) of a " << height << "x" << width << " matrix:" << std::endl;
    for (int y = 0; y < 10 && y < height; ++y) {
        for (int x = 0; x < 10 && x < width; ++x) {
            std::cout << mat[y * width + x].x << " | ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Parameters: d_input, d_output, width, height
void reorganizeSubBlocksTest(cuFloatComplex* h_input, cuFloatComplex* h_output, cuFloatComplex* d_input, cuFloatComplex* d_output, int width, int height) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(width / BLOCK_SIZE, height / BLOCK_SIZE);
    int matrixSize = width * height;
    std::cout << "\nInput Matrix:" << std::endl;
    displayMatrix10x10(h_input, width, height);
    reorganizeSubblocks<<<gridDim, blockDim>>>(d_input, d_output, width, height, true);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    reorganizeSubblocks<<<gridDim, blockDim>>>( d_output, d_input, width, height, false);
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    CUDA_CHECK_ERROR(cudaMemcpy(h_input, d_input, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK_ERROR(cudaMemcpy(h_output, d_output, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    std::cout << "\nInput Matrix:" << std::endl;
    displayMatrix10x10(h_input, width, height);
    std::cout << "\nReorganized Matrix:" << std::endl;
    displayMatrix10x10(h_output, width, height);
}


int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./file.exe <height> <width>" << std::endl;
        return -1;
    }

    const int height = atoi(argv[1]);
    const int width = atoi(argv[2]);

    if (width % BLOCK_SIZE != 0 || height % BLOCK_SIZE != 0) {
        std::cerr << "Width and Height must be divisible by " << BLOCK_SIZE << std::endl;
        return -1;
    }

    // Retrieving device properties
    #if DEBUG
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "--- Device Properties ---" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Max threads per block dimension: " << prop.maxThreadsDim[0] << "x" << prop.maxThreadsDim[1] << "x" << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid size: " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << std::endl;
    std::cout << "Number of stream multiprocessor: " << prop.multiProcessorCount << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;
    std::cout << "Max blocks per multiprocessor: " << prop.maxBlocksPerMultiProcessor << std::endl;
    std::cout << "--------------------------\n";
    #endif

    std::cout << "Matrix dimensions: " << height << "x" << width << std::endl;


    size_t matrixSize = width * height;

    // Host memory allocation
    std::unique_ptr<cuFloatComplex[]> h_input(new cuFloatComplex[matrixSize]);
    std::unique_ptr<cuFloatComplex[]> h_output(new cuFloatComplex[matrixSize]);
    std::unique_ptr<cuFloatComplex[]> h_cufft_output(new cuFloatComplex[matrixSize]);

    // Device memory allocation
    cuFloatComplex *d_input, *d_output;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_input, matrixSize * sizeof(cuFloatComplex)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_output, matrixSize * sizeof(cuFloatComplex)));
    // cuFFT device memory
    cuFloatComplex *d_cufft_input, *d_cufft_output, *d_cufft_quantized;
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cufft_input, matrixSize * sizeof(cuFloatComplex)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cufft_output, matrixSize * sizeof(cuFloatComplex)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_cufft_quantized, matrixSize * sizeof(cuFloatComplex)));

    std::chrono::duration<double> custom_fft_time, cufft_time;

    // Determine batch size for cuFFT
    int fft_size_2d[] = {BLOCK_SIZE, BLOCK_SIZE};
    int batch = (width / BLOCK_SIZE) * (height / BLOCK_SIZE);

    // Kernel launch parameters
    dim3 blockDim
    (BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(width / BLOCK_SIZE, height / BLOCK_SIZE);


    // First iteration is not timed to avoid cold start
    for (unsigned iterToTime = 0; iterToTime < iTT+1; iterToTime++) {

        // Initialize random data
        generateData<123>(h_input.get(), width, height);
        
        // Copy data to device
        CUDA_CHECK_ERROR(cudaMemcpy(d_input, h_input.get(), matrixSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

        #if DEBUG
        // Test reorganizeSubblocks
        reorganizeSubBlocksTest(h_input.get(), h_output.get(), d_input, d_output, width, height);
        #endif

        // Reorganize data to have 8x8 submatrices stored contiguously in memory
        reorganizeSubblocks<<<gridDim, blockDim>>>(d_input, d_cufft_input, width, height, true);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_ERROR(cudaMemcpy(d_input, d_cufft_input, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToDevice));
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // Launch the custom FFT kernel
        auto start = std::chrono::high_resolution_clock::now();
        fftQuantizeKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        // Check for errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }
        
        // Reorganize data back to original layout and copy to host
        // TODO: Reorganize data in-place
        reorganizeSubblocks<<<gridDim, blockDim>>>(d_output, d_input, width, height, false);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_ERROR(cudaMemcpy(h_output.get(), d_input, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

        // Timing
        if (iterToTime > 0)
            custom_fft_time += end - start;

        /*
        --------------------------
            cuFFT Implementation
        --------------------------
        */
        #if USE_CUFFT
        // Create cuFFT plan for batched 8x8 FFTs
        cufftHandle plan;
        if (cufftPlanMany(&plan,
                        2,              // Dimensionality of the transform
                        fft_size_2d,       // Size of each dimension
                        NULL,        // If set to NULL all other advanced data layout parameters are ignored.
                        1,           // Distance between two successive input elements in the least significant (i.e., innermost) dimension.
                        0,         // Distance between the first element of two consecutive signals in a batch of the input data.
                        NULL,        // If set to NULL all other advanced data layout parameters are ignored.
                        1,           // ostride
                        0, // distance between the first element of two consecutive signals in a batch of the output data
                        CUFFT_C2C,
                        batch) != CUFFT_SUCCESS) {
            std::cerr << "CUFFT Error: Unable to create plan" << std::endl;
            return EXIT_FAILURE;
        }

        // Execute cuFFT
        start = std::chrono::high_resolution_clock::now();
        // Execute cuFFT plan
        if (cufftExecC2C(plan, d_cufft_input, d_cufft_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::cerr << "CUFFT Error: Unable to execute plan" << std::endl;
            return EXIT_FAILURE;
        }
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        
        // Launch quantization kernel
        quantizeKernel<<<gridDim, blockDim>>>(d_cufft_output, d_cufft_quantized, width, height);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());

        // Error checking for quantization kernel
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Quantization Kernel Launch Error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

        // Timing
        end = std::chrono::high_resolution_clock::now();        
        if (iterToTime)
            cufft_time += end - start;

        // Reorganize data back to original layout and copy quantized cuFFT results to host
        // TODO: Reorganize data in-place
        reorganizeSubblocks<<<gridDim, blockDim>>>(d_cufft_quantized, d_cufft_output, width, height, false);
        CUDA_CHECK_ERROR(cudaDeviceSynchronize());
        CUDA_CHECK_ERROR(cudaMemcpy(h_cufft_output.get(), d_cufft_output, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
        
        // Destroy cuFFT plan
        cufftDestroy(plan);

        /*
        // Apply quantization to cuFFT results
        for (int by = 0; by < height; by += BLOCK_SIZE) {
            for (int bx = 0; bx < width; bx += BLOCK_SIZE) {
                for (int y = 0; y < BLOCK_SIZE; ++y) {
                    for (int x = 0; x < BLOCK_SIZE; ++x) {
                        int idx = (by + y) * width + (bx + x);
                        int qIdx = y * BLOCK_SIZE + x;
                        float quant = quantizationMatrix[qIdx];
                        float magnitude = cuCabsf(m_cufft_output[idx]) / quant;
                        m_cufft_output[idx] = make_cuFloatComplex(magnitude, 0);
                    }
                }
            }
        }
        */

        // Compare the results
        bool match = true;
        float epsilon = 1e-3f;
        constexpr size_t maxCheckSize = 400;
        for (size_t i = 0; i < matrixSize && i < maxCheckSize; ++i) {
            // Absolute difference
            float abs_diff = cuCabsf(cuCsubf(h_output[i], h_cufft_output[i]));
            if (abs_diff > epsilon) {
                match = false;
                #if DEBUG
                std::cout << "\nMismatch at index " << i << ", abs difference: " << abs_diff;
                std::cout << "\ncuFFT result: (" << h_cufft_output[i].x << ", " << h_cufft_output[i].y << ")";
                std::cout << "\nKernel result: (" << h_output[i].x << ", " << h_output[i].y << ")\n\n";
                #endif
                break;
            }
        }

        if (match) {
            #if DEBUG
            if (matrixSize > maxCheckSize)
                std::cout << "\nResults SEEMS to match with cuFFT implementation.\n\n";
            else
                std::cout << "\nResults match with cuFFT implementation.\n\n";
            #endif
        } else {
            std::cout << "\nResults DO NOT match with cuFFT implementation.\n";
            #if DEBUG
                std::cout << "cuFFT output matrix:\n";
                displayMatrix10x10(h_cufft_output.get(), width, height);
            #endif
        }
        #endif // if USE_CUFFT
    }

    std::cout << "\nCustom FFT Kernel Time: " << custom_fft_time.count()/iTT << " seconds on average over " << iTT << " iterations.\n";
    
    #if USE_CUFFT
    std::cout << "cuFFT with Quantization Time: " << cufft_time.count()/iTT << " seconds on average over " << iTT << " iterations.\n";
    #endif

    // Display the resulting matrix
    std::cout << "\nInput Matrix:" << std::endl;
    displayMatrix10x10(h_input.get(), width, height);
    std::cout << "Cuda kernel output matrix:\n";
    displayMatrix10x10(h_output.get(), width, height);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_cufft_input);
    cudaFree(d_cufft_output);
    cudaFree(d_cufft_quantized);
    

    return 0;
}