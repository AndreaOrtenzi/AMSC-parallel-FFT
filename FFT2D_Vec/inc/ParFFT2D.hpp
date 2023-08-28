
#include <complex>
#include <omp.h>
#include <vector>

class ParFFT2D  {
public: 
  
    template <class C> 
    static void trasform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix);

    template <class C> 
    static void iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix);

};

#include "../src/ParFFT2D.tpp"