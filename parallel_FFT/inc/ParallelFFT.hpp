#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>


#include "AbstractFFT.hpp"

class ParallelFFT : public AbstractFFT {
public:
    // using AbstractFFT::AbstractFFT;
    ParallelFFT(const std::vector<std::complex<real>>& sValues, const std::vector<std::complex<real>>& fValues) : AbstractFFT(sValues, fValues){} // inherit constructor

    const std::vector<std::complex<real>>& getSpatialValues() const override;

    const std::vector<std::complex<real>>& getFrequencyValues() const override;

    void transform() override;

    void iTransform() override;

private:
    void iterativeFFT(std::complex<real> x[], const unsigned int n);
};