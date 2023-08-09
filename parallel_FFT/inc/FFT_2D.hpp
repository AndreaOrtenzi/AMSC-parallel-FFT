#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra> 

// Useful:
using namespace std;
using namespace Eigen;

using SpVec = Eigen::VectorXcd;
using SpMat = Eigen::SparseMatrix<complex>;

class FFT_2D 
{
    public:
    FFT_2D() {};

    /*To generate input matrix, the matrix should have as number of rows and columns a power of 2.
    For this reason, we set a method taking as input the power. We choose to fill randomly the matrix with real and imaginary numbers
    that are maximum equal to 250.0: */ 
    void generate_input(usigned int pow);

    // Sequential Iterative implementation:
    void iterative_sequential();

    // Parallel Iterative implementation usign OpenMP library: 
    void iterative_parallel();

    // Using Eigen library, we can even load the input inserting the name of the file as input for this method.
    void load_input(const std::string& filename);

    // Apply inverse transform on spectral matrix:
    void inverse_transform(SpMat& spectral); 

    private:
    SpMat input_matrix; // input matrix
    SpMat iter_seq_sol; //  after that sequential iterative FFT has been applied
    SpMat iter_par_sol; // spectral matrix obtained after that parallel iterative FFT has been applied
    unsigned int n; // matrix's dimension
    SpMat inverse_sol; // matrix obtained after IFFT has been applied on spectral matrix

};
