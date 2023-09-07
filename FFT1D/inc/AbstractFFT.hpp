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
#include <stdexcept> // Added for exception handling

class AbstractFFT {
public:
    // Constructors:

    // Constructor 1: Initializes the AbstractFFT object with a specified problem size.
    // Parameters:
    //   - problemSize: An unsigned integer representing the size of the problem.
    AbstractFFT(const unsigned int problemSize) : N(problemSize) { isPowerOfTwo(N); }

    // Constructor 2: Initializes the AbstractFFT object with spatial and frequency values.
    // Parameters:
    //   - sValues: A vector of std::complex<real> representing spatial values.
    //   - fValues: A vector of std::complex<real> representing frequency values.
    AbstractFFT(const std::vector<std::complex<real>>& sValues, const std::vector<std::complex<real>>& fValues) :
        spatialValues(sValues),
        frequencyValues(fValues),
        N(std::max(sValues.size(), fValues.size())) { isPowerOfTwo(N); }

    // Accessors for spatialValues and frequencyValues:
    const std::vector<std::complex<real>>& getSpatialValues() const { return spatialValues; }
    const std::vector<std::complex<real>>& getFrequencyValues() const { return frequencyValues; }

    // Transform and inverse transform methods:
    void transform() {
        transform(spatialValues);
    }

    void iTransform() {
        iTransform(frequencyValues);
    }

    // Pure virtual methods to be implemented by derived classes:
    virtual void transform(const std::vector<std::complex<real>>& sValues) = 0;
    virtual void iTransform(const std::vector<std::complex<real>>& fValues) = 0;

    // Destructor:
    virtual ~AbstractFFT() {};

protected:
    // Pure virtual methods for FFT implementations:
    virtual void iterativeFFT(std::complex<real> x[], const unsigned int n) = 0;
    virtual void recursiveFFT(std::complex<real> x[], const unsigned int n) = 0;

    // Helper function to check input vector size:
    void checkInputValues(const std::vector<std::complex<real>>& values) const {
        if (values.size() != N)
            throw std::invalid_argument("Array size is not N");
    };

    std::vector<std::complex<real>> spatialValues;
    std::vector<std::complex<real>> frequencyValues;
    const unsigned int N; 

private:
    // Helper function to check if N is a power of 2
    static void isPowerOfTwo(unsigned int n) {
        if (n == 0)
            throw std::invalid_argument("Array size is 0");

        if (ceil(log2(n)) != floor(log2(n)))
            throw std::invalid_argument("Array size is not a power of 2");
    }
};

#endif // ABSTRACT_FFT_HPP
