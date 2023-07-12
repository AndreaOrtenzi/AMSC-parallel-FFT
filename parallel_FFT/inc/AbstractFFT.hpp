#ifndef ABSTRACT_FFT_HPP
#define ABSTRACT_FFT_HPP

#ifndef Real
#define Real double
#endif
using real = Real;


#include <vector>
#include <complex>

class AbstractFFT {
protected:

    AbstractFFT(const std::vector<std::complex<real>>& sValues,const std::vector<std::complex<real>>& fValues) : 
        spatialValues(sValues)
        , frequencyValues(fValues)
        , N(std::max(sValues.size(),fValues.size())) {}

    virtual const std::vector<std::complex<real>>& getSpatialValues() const = 0;
    virtual const std::vector<std::complex<real>>& getFrequencyValues() const = 0;
    virtual void transform() = 0;
    virtual void iTransform() = 0;
    virtual ~AbstractFFT() {};

// protected:
    std::vector<std::complex<real>> spatialValues;
    std::vector<std::complex<real>> frequencyValues;
    const unsigned int N; // vectors' size
};

#endif // ABSTRACT_FFT_HPP