#ifndef ABSTRACT_FFT2D_HPP
#define ABSTRACT_FFT2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra> 
#include <complex>
#include <vector>

// Useful aliases for simplifying code:
using namespace std;
using namespace Eigen;
using Vec = Eigen::VectorXcd;
using Mat = Eigen::MatrixXcd;

// Abstract class for Fast Fourier Transform 2D, designed only for square matrices.
class AbstractFFT2D {
protected:
    // Constructor that takes spatial and frequency matrices
    AbstractFFT2D(const Mat& sValues, const Mat& fValues) : 
        spatialValues(sValues)
        , frequencyValues(fValues)
        , n(sValues.rows()) {}

    // Virtual functions to be implemented by derived classes
    virtual const Mat& getSpatialValues() const = 0;                    // Get spatial matrix
    virtual const Mat& getFrequencyValues() const = 0;                  // Get frequency matrix
    virtual void transform_par(const unsigned int numThreads) = 0;      // Parallel FFT transformation
    virtual void transform_seq() = 0;                                   // Sequential FFT transformation
    virtual void iTransform() = 0;                                      // Inverse FFT transformation
    virtual ~AbstractFFT2D() {};                                        // Destructor

// Member variables
    Mat spatialValues;          // Input spatial matrix
    Mat frequencyValues;        // Output frequency matrix
    const unsigned int n;       // Size of matrices
};

#endif // ABSTRACT_FFT2D_HPP
