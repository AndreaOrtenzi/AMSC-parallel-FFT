#include <vector>
#include <complex>
#include <cmath>
#include <omp.h>


#include "AbstractFFT.hpp"

/*  Potremmo implementare una classe unica per la FFT unidimensionale senza specificare sequential o parallel, quindi magari cambiando
    il nome della classe, così da avere nella stessa classe sia i metodi sequenziali che quello parallelo.*/

class SequentialFFT : public AbstractFFT {
public:
    // using AbstractFFT::AbstractFFT;
    SequentialFFT(const std::vector<std::complex<real>>& sValues, const std::vector<std::complex<real>>& fValues) : AbstractFFT(sValues, fValues){} // inherit constructor

    const std::vector<std::complex<real>>& getSpatialValues() const override;

    const std::vector<std::complex<real>>& getFrequencyValues() const override;

    void transform() override;

    void iTransform() override;

private:
    void recursiveFFT(std::complex<real> x[], const unsigned int n);
    void iterativeFFT(std::complex<real> x[], const unsigned int n);
    void parallelFFT (std::complex<real> x[], const unsigned int n);
    static bool isRecursive;
};