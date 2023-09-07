#include <complex>
// #include <cmath>
#include <vector>

class SeqFFT2D {
public:
    // Perform a sequential 2D Fourier transform on the input matrix.
    // Parameters:
    //   input_matrix: Reference to the input matrix to be transformed.
    //   freq_matrix: Reference to the output frequency domain matrix.
    template <class C>
    static void transform(std::vector<std::vector<C>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix);

    // Perform a sequential inverse 2D Fourier transform on the input frequency domain matrix.
    // Parameters:
    //   input_matrix: Reference to the input frequency domain matrix to be transformed back to space domain.
    //   space_matrix: Reference to the output space domain matrix.
    template <class C>
    static void iTransform(std::vector<std::vector<std::complex<double>>>& input_matrix, std::vector<std::vector<C>>& space_matrix);
};


#include "../src/SeqFFT2D.tpp"
