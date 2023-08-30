#ifndef PARALLEL_OMP_FFT_HPP
#define PARALLEL_OMP_FFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>

#include "AbstractFFT.hpp"

class Parallel_OMP_FFT : public AbstractFFT {
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

private:
    static bool isRecursive;
};

#endif // PARALLEL_OMP_FFT_HPP