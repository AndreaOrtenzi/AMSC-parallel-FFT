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
        for (unsigned int k = 0; k < N; ++k) {
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