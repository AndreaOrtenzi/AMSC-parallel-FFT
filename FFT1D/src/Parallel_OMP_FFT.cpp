#include "../inc/Parallel_OMP_FFT.hpp"
#include <iostream>

bool Parallel_OMP_FFT::isRecursive = false;

// Perform the Fourier transform on the spatial values and store the result in the frequency values
void Parallel_OMP_FFT::transform(const std::vector<std::complex<real>>& sValues) {
    frequencyValues.resize(N);
    frequencyValues = sValues;

    // Check whether to use recursive or iterative implementation
    if (isRecursive){
        std::cout << "--start recursive implementation--" << std::endl;
        recursiveFFT(frequencyValues.data(), frequencyValues.size());
    }
    else {
        std::cout << "--start iterative implementation--" << std::endl;
        iterativeFFT(frequencyValues.data(), frequencyValues.size());
    }

    // Toggle between recursive and iterative for the next transform
    isRecursive = !isRecursive;
}

// A parallel implementation of the FFT recursive method using OpenMP.
void Parallel_OMP_FFT::recursiveFFT(std::complex<real> x[], const unsigned int n) {
    if (n <= 1) {
        return;
    }
    // Create vectors of even and odd indexes:
    std::complex<real> even[n/2], odd[n/2];
    for (unsigned int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }

    // Determine the number of threads to use, try with different values
    //******************************************************************
    // unsigned int numThreads = static_cast<unsigned int>(log2(n));
    // unsigned int numThreads = omp_get_max_threads();
    unsigned int numThreads = 4;
    // unsigned int numThreads = n;
    // ******************************************************************

    // Use OpenMP parallel sections to parallelize FFT computation for even and odd halves
    #pragma omp parallel sections num_threads(numThreads) 
    {
        #pragma omp section
        recursiveFFT(even, n / 2); // FFT on even-indexed elements

        #pragma omp section
        recursiveFFT(odd, n / 2); // FFT on odd-indexed elements
    }

    // Combine the results of the two subproblems in parallel
    #pragma omp parallel for schedule(static)
    for (unsigned int i = 0; i < n / 2; i++) {
        std::complex<real> t((std::complex<real>)std::polar(1.0, -2 * M_PI * i / n) * odd[i]);
        x[i] = even[i] + t;
        x[i + n / 2] = even[i] - t;
    }
}

// A parallel implementation of the FFT iterative method using OpenMP.
void Parallel_OMP_FFT::iterativeFFT(std::complex<real> x[], const unsigned int n) {
    unsigned int numBits = static_cast<unsigned int>(log2(n));

    //Determine the number of threads to use, try with different values
    //******************************************************************
    // unsigned int numThreads = static_cast<unsigned int>(log2(n));
    // unsigned int numThreads = omp_get_max_threads();
    unsigned int numThreads = 4;
    // unsigned int numThreads = n;
    // ******************************************************************

    // Bit-reversal
    for (unsigned int i = 0; i < n; i++) 
    {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((i >> k) & 1U);
        }
        if (j > i) {
            std::swap(x[i], x[j]);
        }
    }
    
    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<real> wm = std::exp(-2.0 * M_PI * std::complex<real>(0, 1) / static_cast<real>(m));
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for (unsigned int k = 0; k < n; k += m) {
            std::complex<real> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<real> t = w * x[k + j + m / 2];
                std::complex<real> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
}

// Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
void Parallel_OMP_FFT::iTransform(const std::vector<std::complex<real>>& fValues) {
    spatialValues.resize(N);

    //Determine the number of threads to use, try with different values
    //******************************************************************
    // unsigned int numThreads = static_cast<unsigned int>(log2(n));
    // unsigned int numThreads = omp_get_max_threads();
    unsigned int numThreads = 4;
    // unsigned int numThreads = n;
    // ******************************************************************

    std::vector<std::complex<real>> freqVec = fValues;
    unsigned int numBits = static_cast<unsigned int>(log2(N));

    // Bit reversal
    #pragma omp parallel for num_threads(numThreads) schedule(static)
    for (unsigned int l = 0; l < N; l++) {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((l >> k) & 1U);
        }
        if (j > l) {
            std::swap(freqVec[l], freqVec[j]);
        }
    }
    
    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for (unsigned int k = 0; k < N; k += m) {
            std::complex<double> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<double> t = w * freqVec[k + j + m / 2];
                std::complex<double> u = freqVec[k + j];
                freqVec[k + j] = u + t;
                freqVec[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    // Normalize by dividing by N
    for (unsigned int i = 0; i < N; ++i) {
        spatialValues[i] = freqVec[i] / static_cast<real>(N);
    }
}
