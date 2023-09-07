
#include <complex>
#include <omp.h>
#include <vector>

#include <unordered_map>

class ParFFT2D {
public:
    // Perform a 2D Fourier transform on the input matrix in parallel.
    // Parameters:
    //   input_matrix: Reference to the input matrix to be transformed.
    //   freq_matrix: Reference to the output frequency domain matrix.
    //   n_threads: The number of threads to use for parallel processing.
    template <class C>
    static void transform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix, const unsigned int n_threads);

    // Perform an inverse 2D Fourier transform on the input frequency domain matrix in parallel.
    // Parameters:
    //   input_matrix: Reference to the input frequency domain matrix to be transformed back to space domain.
    //   space_matrix: Reference to the output space domain matrix.
    //   n_threads: The number of threads to use for parallel processing.
    template <class C>
    static void iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix, const unsigned int n_threads);
};


#include "../src/ParFFT2D.tpp"