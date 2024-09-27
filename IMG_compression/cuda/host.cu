#include <iostream>
#include <memory>
#include <random>
#include <cufft.h>
#include <chrono>

#ifndef VERSION
#define VERSION 1
#endif
#ifndef iTT
#define iTT 1
#endif

#if VERSION == 1
#include "device_v1.cu"
#elif VERSION == 2
#include "device_v2.cu"
#elif VERSION == 3
#include "device_v3.cu"
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
void generateData(cuFloatComplex* data, size_t matrixSize) {
    if constexpr (DATA == 0) {
        for (int i = 0; i < matrixSize; ++i) {
            data[i] = make_cuFloatComplex(0., 0.);;
        }
    } else if constexpr (DATA == 1) {
        for (int i = 0; i < matrixSize; ++i) {
            data[i] = make_cuFloatComplex(3., 0.);;
        }
    } else if constexpr (DATA == 123) {
        std::mt19937 gen(0); // Fixed seed for reproducibility
        std::uniform_int_distribution<int> dis(0, 255);

        for (size_t i = 0; i < matrixSize; ++i) {
            float val = static_cast<float>(dis(gen));
            data[i] = make_cuFloatComplex(val, 0);
        }
    }
}

void displayMatrix10x10(cuFloatComplex* mat, int width, int height) {
    std::cout << "Top-left 10x10 corner (real part):" << std::endl;
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
        std::cout << "Usage: ./file.exe <width> <height>" << std::endl;
        return -1;
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);

    if (width % BLOCK_SIZE != 0 || height % BLOCK_SIZE != 0) {
        std::cerr << "Width and Height must be divisible by " << BLOCK_SIZE << std::endl;
        return -1;
    }
    std::cout << "Matrix dimensions: " << width << "x" << height << std::endl;

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
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(width / BLOCK_SIZE, height / BLOCK_SIZE);

    for (unsigned iterToTime = 0; iterToTime < iTT; iterToTime++) {

        // Initialize random data
        generateData<123>(h_input.get(), matrixSize);
        
        // Copy data to device
        CUDA_CHECK_ERROR(cudaMemcpy(d_input, h_input.get(), matrixSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
        CUDA_CHECK_ERROR(cudaMemcpy(d_cufft_input, h_input.get(), matrixSize * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

        // Test reorganizeSubblocks
        #if DEBUG
        reorganizeSubBlocksTest(h_input.get(), h_output.get(), d_input, d_output, width, height);
        #endif

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

        // Copy results back to host
        CUDA_CHECK_ERROR(cudaMemcpy(h_output.get(), d_output, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

        // Timing
        custom_fft_time += end - start;

        /*
        --------------------------
            cuFFT Implementation
        --------------------------
        */
        // Create cuFFT plan for batched 8x8 FFTs
        cufftHandle plan;
        int inembed[] = {width, BLOCK_SIZE};
        if (cufftPlanMany(&plan,
                        2,              // Dimensionality of the transform
                        fft_size_2d,       // Size of each dimension
                        NULL,        // If set to NULL all other advanced data layout parameters are ignored.
                        1,           // Distance between two successive input elements in the least significant (i.e., innermost) dimension.
                        0,         // Distance between the first element of two consecutive signals in a batch of the input data.
                        NULL,        // If set to NULL all other advanced data layout parameters are ignored.
                        1,           // ostride
                        8, // distance between the first element of two consecutive signals in a batch of the output data
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

        end = std::chrono::high_resolution_clock::now();

        // Copy quantized cuFFT results back to host
        cudaMemcpy(h_cufft_output.get(), d_cufft_quantized, matrixSize * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
        cufftDestroy(plan);

        // Timing
        cufft_time += end - start;

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
        for (size_t i = 0; i < matrixSize; ++i) {
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
            std::cout << "\nResults match with cuFFT implementation.\n\n";
        } else {
            std::cout << "\nResults DO NOT match with cuFFT implementation.\n";
            #if DEBUG
                std::cout << "cuFFT output matrix:\n";
                displayMatrix10x10(h_cufft_output.get(), width, height);
            #endif
        }
    }

    std::cout << "\nCustom FFT Kernel Time: " << custom_fft_time.count()/iTT << " seconds on average over " << iTT << " iterations.\n";
    std::cout << "cuFFT with Quantization Time: " << cufft_time.count()/iTT << " seconds on average over " << iTT << " iterations.\n";

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
