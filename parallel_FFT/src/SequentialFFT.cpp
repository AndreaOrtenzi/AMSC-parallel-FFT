#include "../inc/SequentialFFT.hpp"
#include <iostream>

bool SequentialFFT::isRecursive = false;

const std::vector<std::complex<real>>& SequentialFFT::getSpatialValues() const {
    return spatialValues;
}

const std::vector<std::complex<real>>& SequentialFFT::getFrequencyValues() const {
    return frequencyValues;
}

void SequentialFFT::transform() {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values
    frequencyValues.resize(N);
    if (isRecursive){
        std::cout << "--start recursive imp--" << std::endl;
        frequencyValues = spatialValues;
        recursiveFFT(frequencyValues.data(),frequencyValues.size());
    }
    else{
        std::cout << "--start iterative imp--" << std::endl;
        frequencyValues = spatialValues;
        iterativeFFT(frequencyValues.data(),frequencyValues.size());
    }
    isRecursive = !isRecursive;
}

// A recursive implementation for FFT
void SequentialFFT::recursiveFFT(std::complex<real> x[], const unsigned int n) {
    if (n <= 1) {
        return;
    }
    std::complex<real> even[n/2], odd[n/2];
    for (unsigned int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }
    recursiveFFT(even, n/2);
    recursiveFFT(odd, n/2);
    for (unsigned int i = 0; i < n/2; i++) {
        std::complex<real> t((std::complex<real>)std::polar(1.0, -2*M_PI*i/n) * odd[i]);
        x[i] = even[i] + t;
        x[i+n/2] = even[i] - t;
    }
}

// An iterative implementation for FFT.
void SequentialFFT::iterativeFFT(std::complex<real> x[], const unsigned int n) {
    unsigned int numBits = static_cast<unsigned int>(log2(n));
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


void SequentialFFT::iTransform() {
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    spatialValues.resize(N);
    for (unsigned int n = 0; n < N; ++n) {
        std::complex<real> sum(0, 0);
        for (unsigned int k = 0; k < N; ++k) {
            std::complex<real> term = frequencyValues[k] * std::exp(2.0 * M_PI * std::complex<real>(0, 1) * static_cast<real>(k * n) / static_cast<real>(N));
            sum += term;
        }
        spatialValues[n] = sum / static_cast<real>(N);
    }
}