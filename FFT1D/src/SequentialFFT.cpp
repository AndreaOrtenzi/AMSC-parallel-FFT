#include "../inc/SequentialFFT.hpp"
#include <iostream>

bool SequentialFFT::isRecursive = false;

void SequentialFFT::transform(const std::vector<std::complex<real>>& sValues) {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values
    frequencyValues.resize(N);
    // Check whether to use recursive or iterative implementation
    if (isRecursive){
        std::cout << "--start recursive imp--" << std::endl;
        frequencyValues = sValues;
        recursiveFFT(frequencyValues.data(),frequencyValues.size());
    }
    else{
        std::cout << "--start iterative imp--" << std::endl;
        frequencyValues = sValues;
        iterativeFFT(frequencyValues.data(),frequencyValues.size());
    }
    isRecursive = !isRecursive;
}

// A recursive implementation for FFT
void SequentialFFT::recursiveFFT(std::complex<real> x[], const unsigned int n) {
    if (n <= 1) {
        return;
    }

    // Divide the input into even and odd halves
    std::complex<real> even[n/2], odd[n/2];
    for (unsigned int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }

    // Recursively calculate FFT for even and odd halves
    recursiveFFT(even, n/2);
    recursiveFFT(odd, n/2);
    // Combine results from even and odd halves using twiddle factors:
    for (unsigned int i = 0; i < n/2; i++) {
        std::complex<real> t((std::complex<real>)std::polar(1.0, -2*M_PI*i/n) * odd[i]);
        x[i] = even[i] + t;
        x[i+n/2] = even[i] - t;
    }
}

// An iterative implementation for FFT.
void SequentialFFT::iterativeFFT(std::complex<real> x[], const unsigned int n) {
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    
    // Bit-reversal:
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

// An iterative iFFT version:
void SequentialFFT::iTransform(const std::vector<std::complex<real>>& fValues) {
    
    unsigned int n = fValues.size();
    std::vector<std::complex<real>> freqVec = fValues;
    // Bit-reversal:
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    for (unsigned int i = 0; i < n; i++) 
    {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((i >> k) & 1U);
        }
        if (j > i) {
            std::swap(freqVec[i], freqVec[j]);
        }
    }
    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<real> wm = std::exp(2.0 * M_PI * std::complex<real>(0, 1) / static_cast<real>(m));
        for (unsigned int k = 0; k < n; k += m) {
            std::complex<real> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<real> t = w * freqVec[k + j + m / 2];
                std::complex<real> u = freqVec[k + j];
                freqVec[k + j] = u + t;
                freqVec[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    // Real coefficient 1/N normalization:
    real N_inv = 1.0 / static_cast<real>(n);

    spatialValues.resize(n);
    for (unsigned int i = 0; i < n; i++) {
        spatialValues[i] = freqVec[i] * N_inv;
    }
}

