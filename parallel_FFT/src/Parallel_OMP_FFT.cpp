#include "../inc/Parallel_OMP_FFT.hpp"
#include <iostream>

bool Parallel_OMP_FFT::isRecursive = false;

void Parallel_OMP_FFT::transform(const std::vector<std::complex<real>>& sValues) {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values
    frequencyValues.resize(N);
    if (isRecursive){
        std::cout << "--start recursive imp--" << std::endl;
        frequencyValues = sValues;
        recursiveFFT(frequencyValues.data(),frequencyValues.size());
    }
    else {
        std::cout<< "--start iterative imp--" << std::endl;
        frequencyValues = sValues;
        iterativeFFT(frequencyValues.data(),frequencyValues.size());
    }

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

        //******************************************************************
    //          Try with different numbers of threads 
    // unsigned int numThreads = static_cast<unsigned int>(ceil(log2(n)));
    unsigned int numThreads = 2;
    // unsigned int numThreads = 4;
    // unsigned int numThreads = n;
    // ******************************************************************

    #pragma omp parallel sections num_threads(numThreads) 
    {
        #pragma omp section
        recursiveFFT(even, n / 2); // FFT on even-indexed elements

        #pragma omp section
        recursiveFFT(odd, n / 2); // FFT on odd-indexed elements
    }

    // Combine the results of the two subproblems:
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

    //******************************************************************
    //          Try with different numbers of threads 
    // unsigned int numThreads = static_cast<unsigned int>(ceil(log2(n)));
    unsigned int numThreads = 2;
    // unsigned int numThreads = 4;
    // unsigned int numThreads = n;
    // ******************************************************************

    // Create region of parallel tasks in order to do bit reverse for input vector x, n is shared among all the threads of the region:
    #pragma omp task shared(x) firstprivate(n)
    #pragma omp parallel for num_threads(numThreads) schedule(static)
    for (unsigned int i = 0; i < n; i++) {
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
        std::complex<real> wm = std::exp(-2.0 * M_PI * std::complex<real>(0, 1) / static_cast<real>(m)); // Twiddle factor
    
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


void ParallelFFT::iTransform(const std::vector<std::complex<real>>& fValues) {
    //Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    spatialValues.resize(N);

    unsigned int numThreads = static_cast<unsigned int> (ceil(log2(N)));
    std::vector<std::complex<real>> thread_partialsums(N * numThreads, std::complex<real>(0, 0));

    #pragma omp parallel num_threads(numThreads)
    {
        unsigned int tid = omp_get_thread_num();
        for (unsigned int n = 0; n < N; ++n) {
            std::complex<real> sum(0, 0);
            for (unsigned int k = 0; k < N; ++k) {
                std::complex<real> term = fValues[k] * std::exp(2.0 * M_PI * std::complex<real>(0, 1) * static_cast<real>(k * n) / static_cast<real>(N));
                sum += term;
            }
            thread_partialsums[tid * N + n] = sum;
        }
    }

    //Combine partial sums from different threads
    for (unsigned int n = 0; n < N; ++n) {
        std::complex<real> sum(0, 0);
        for (unsigned int t = 0; t < numThreads; t++) {
            sum += thread_partialsums[t * N + n];
        }
        spatialValues[n] = sum / static_cast<real>(N);
    }
}