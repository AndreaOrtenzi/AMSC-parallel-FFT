#ifndef SEQUENTIAL_FFT_HPP
#define SEQUENTIAL_FFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>

#include "AbstractFFT.hpp"

class SequentialFFT : public AbstractFFT {
public:
    using AbstractFFT::AbstractFFT; // use AbstractFFT constructors
    // use AbstractFFT overloaded methods, overloading hides the parent's methods
    using AbstractFFT::transform;
    using AbstractFFT::iTransform;

    void transform(const std::vector<std::complex<real>>& sValues) override;
    void iTransform(const std::vector<std::complex<real>>& fValues) override;

protected:
    void iterativeFFT(std::complex<real> x[], const unsigned int n) override;
    void recursiveFFT(std::complex<real> x[], const unsigned int n) override;

protected:
    static bool isRecursive; // Static variable to control the FFT method (iterative or recursive)
};

#endif // PARALLEL_FFT_HPP