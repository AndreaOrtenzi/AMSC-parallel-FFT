#include "../inc/ParallelFFT.hpp"
#include <iostream>

bool ParallelFFT::isRecursive = false;

const std::vector<std::complex<real>>& ParallelFFT::getSpatialValues() const {
    return spatialValues;
}

const std::vector<std::complex<real>>& ParallelFFT::getFrequencyValues() const {
    return frequencyValues;
}

void ParallelFFT::transform() {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values
    frequencyValues.resize(N);
    if (isRecursive){
        std::cout << "--start recursive imp--" << std::endl;
        frequencyValues = spatialValues;
        recursiveFFT(frequencyValues.data(),frequencyValues.size());
    }
    else {
        unsigned int k;
        #pragma omp parallel for private(k) shared(spatialValues, frequencyValues)
        for (k = 0; k < N; ++k) {
            std::complex<real> sum(0, 0);
            for (unsigned int n = 0; n < N; ++n) {
                std::complex<real> term = spatialValues[n] * std::exp(-2.0 * M_PI * std::complex<real>(0, 1) * static_cast<real>(k * n) / static_cast<real>(N));
                sum += term;
            }
            frequencyValues[k] = sum;
        }
    }

    isRecursive = !isRecursive;
}

// // A parallel implementation of the FFT recursive method using OpenMP.
void ParallelFFT::recursiveFFT(std::complex<real> x[], const unsigned int n) {
    if (n <= 1) {
        return;
    }

    // unsigned int numThreads = omp_get_max_threads();

    #pragma omp parallel sections num_threads(2)
    {
        #pragma omp section
        recursiveFFT(x, n / 2); // FFT on even-indexed elements

        #pragma omp section
        recursiveFFT(x + n / 2, n / 2); // FFT on odd-indexed elements
    }

    // Combine the results of the two subproblems
    #pragma omp parallel for
    for (unsigned int i = 0; i < n / 2; i++) {
        std::complex<real> t((std::complex<real>)std::polar(1.0, -2 * M_PI * i / n) * x[i + n / 2]);
        x[i] = x[i] + t;
        x[i + n / 2] = x[i] - t;
    }
}

// A parallel implementation of the FFT iterative method using OpenMP.
void SequentialFFT::parallelFFT(std::complex<real> x[], const unsigned int n){
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    unsigned int numThreads = static_cast<unsigned int> ceil(log2(n));

    #pragma omp parallel for num_threads(numThreads)
    for (unsigned int i = 0; i < n; i++) {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((i >> k) & 1U);
        }
        if (j > i) {
            std::swap(x[i], x[j]);
        }
    }

    #pragma omp parallel num_threads(numThreads)
    {
        #pragma omp for
        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<real> wm = std::exp(-2.0 * M_PI * std::complex<real>(0, 1) / static_cast<real>(m));

            #pragma omp for
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
}


void ParallelFFT::iTransform() {
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    spatialValues.resize(N);

    unsigned int numThreads = static_cast<unsigned int> ceil(log2(n)); ; // Set as desired number of threads the nearest integer number of log2(n)

    #pragma omp parallel for num_threads(numThreads)
    for (unsigned int n = 0; n < N; ++n) {
        std::complex<real> sum(0, 0);

        #pragma omp parallel for reduction(+:sum)
        for (unsigned int k = 0; k < N; ++k) {
            std::complex<real> term = frequencyValues[k] * std::exp(2.0 * M_PI * std::complex<real>(0, 1) * static_cast<real>(k * n) / static_cast<real>(N));
            sum += term;
        }

        spatialValues[n] = sum / static_cast<real>(N);
    }
}