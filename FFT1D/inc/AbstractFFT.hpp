#ifndef ABSTRACT_FFT_HPP
#define ABSTRACT_FFT_HPP

#ifndef Real
#define Real double
#endif
using real = Real;

#ifndef IS_RECURSIVE
#define IS_RECURSIVE false
#endif

#include <vector>
#include <complex>
#include <bits/stdc++.h>

class AbstractFFT {
public:

    AbstractFFT(const unsigned int problemSize) : N(problemSize) {isPowerOfTwo(N);}

    AbstractFFT(const std::vector<std::complex<real>>& sValues,const std::vector<std::complex<real>>& fValues) : 
        spatialValues(sValues)
        , frequencyValues(fValues)
        , N(std::max(sValues.size(),fValues.size())) {isPowerOfTwo(N);}

    const std::vector<std::complex<real>>& getSpatialValues() const {return spatialValues;};
    const std::vector<std::complex<real>>& getFrequencyValues() const {return frequencyValues;};

    void transform() {
        transform(spatialValues);
    }

    void iTransform() {
        iTransform(frequencyValues);
    }

    virtual void transform(const std::vector<std::complex<real>>& sValues) = 0;
    virtual void iTransform(const std::vector<std::complex<real>>& fValues) = 0;
    
    virtual ~AbstractFFT() {};

protected:

    virtual void iterativeFFT(std::complex<real> x[], const unsigned int n) = 0;
    virtual void recursiveFFT(std::complex<real> x[], const unsigned int n) = 0;
    
    void checkInputValues(const std::vector<std::complex<real>>& values) const {
        if (values.size() != N)
            throw std::invalid_argument( "Array size is not N" );
    };

    std::vector<std::complex<real>> spatialValues;
    std::vector<std::complex<real>> frequencyValues;
    const unsigned int N; // vectors' size

private: 
    static void isPowerOfTwo (unsigned int n) {
        if (n == 0)
            throw std::invalid_argument( "Array size is 0" );
    
        if (ceil(log2(n)) != floor(log2(n)))
            throw std::invalid_argument( "Array size is not a power of 2" ); 
    }
};

#endif // ABSTRACT_FFT_HPP