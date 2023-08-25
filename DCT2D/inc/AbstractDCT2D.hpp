#ifndef ABSTRACT_DCT2D_HPP
#define ABSTRACT_DCT2D_HPP

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra> 
#include <complex>
#include <vector>

// Useful:
using namespace std;
using namespace Eigen;
using Vec = Eigen::VectorXi;
using Mat = Eigen::MatrixXi;

class AbstractDCT2D {
protected:

    AbstractDCT2D(const Mat& sValues,const Mat& fValues) : 
        spatialValues(sValues)
        , frequencyValues(fValues)
        , n(std::max( std::max(sValues.rows(), sValues.cols()) , std::max(fValues.rows(),fValues.cols()) ) ) {}

    virtual const Mat& getSpatialValues() const = 0;
    virtual const Mat& getFrequencyValues() const = 0;
    virtual void transform_par() = 0;
    virtual void transform_seq() = 0;
    virtual void iTransform() = 0;
    virtual ~AbstractDCT2D() {};

// protected:
    Mat spatialValues;
    Mat frequencyValues;
    const unsigned int n; // matrices' size
};

#endif // ABSTRACT_DCT2D_HPP