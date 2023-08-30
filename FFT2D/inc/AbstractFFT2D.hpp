#ifndef ABSTRACT_FFT2D_HPP
#define ABSTRACT_FFT2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra> 
#include <complex>
#include <vector>

// Useful:
using namespace std;
using namespace Eigen;
using Vec = Eigen::VectorXcd;
using Mat = Eigen::MatrixXcd;

class AbstractFFT2D {
protected:

    AbstractFFT2D(const Mat& sValues,const Mat& fValues) : 
        spatialValues(sValues)
        , frequencyValues(fValues)
        , n(std::max( std::max(sValues.rows(), sValues.cols()) , std::max(fValues.rows(),fValues.cols()) ) ) {}

    virtual const Mat& getSpatialValues() const = 0;
    virtual const Mat& getFrequencyValues() const = 0;
    virtual void transform_par(const unsigned int numThreads) = 0;
    virtual void transform_seq() = 0;
    virtual void iTransform() = 0;
    virtual ~AbstractFFT2D() {};

// protected:
    Mat spatialValues;
    Mat frequencyValues;
    const unsigned int n; // matrices' size
};

#endif // ABSTRACT_FFT2D_HPP