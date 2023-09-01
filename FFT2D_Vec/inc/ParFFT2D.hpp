
#include <complex>
#include <omp.h>
#include <vector>

#include <unordered_map>

class ParFFT2D  {
public:
  
    template <class C> 
    static void trasform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix, const unsigned int n_threads);

    template <class C> 
    static void iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix, const unsigned int n_threads);

};

#include "../src/ParFFT2D.tpp"