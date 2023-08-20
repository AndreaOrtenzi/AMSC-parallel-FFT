#ifndef ABSTRACT_FFT2D_HPP
#define ABSTRACT_FFT2D_HPP

// #ifndef Real
// #define Real double
// #endif
// using real = Real;

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra> 
#include <complex>
#include <vector>

// Useful:
using namespace std;
using namespace Eigen;
//using MyComplex = std::complex<double>;
using SpVec = Eigen::VectorXcd;
using SpMat = Eigen::MatrixXcd;

class AbstractFFT2D {
protected:

    AbstractFFT2D(const SpMat& sValues,const SpMat& fValues) : 
        spatialValues(sValues)
        , frequencyValues(fValues)
        , n(std::max( std::max(sValues.rows(), sValues.cols()) , std::max(fValues.rows(),fValues.cols()) ) ) {}

    virtual const SpMat& getSpatialValues() const = 0;
    virtual const SpMat& getFrequencyValues() const = 0;
    virtual void transform() = 0;
    virtual void iTransform() = 0;
    virtual ~AbstractFFT2D() {};

// protected:
    SpMat spatialValues;
    SpMat frequencyValues;
    const unsigned int n; // matrices' size
};

#endif // ABSTRACT_FFT2D_HPP