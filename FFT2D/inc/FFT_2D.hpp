#include "AbstractFFT2D.hpp"  // Include the abstract base class

#include <complex>
#include <cmath>
#include <vector>

// Class definition for the Fast Fourier Transform 2D, derived from AbstractFFT2D
class FFT_2D : public AbstractFFT2D
{
public:
    // Constructor that initializes the base class
    FFT_2D(const Mat& sValues, const Mat& fValues) : AbstractFFT2D(sValues, fValues) {};

    const Mat& getSpatialValues() const override;               // Get spatial matrix
    const Mat& getFrequencyValues() const override;             // Get frequency matrix
    void transform_par(const unsigned int numThreads) override; // Parallel FFT transformation
    void transform_seq() override;                              // Sequential FFT transformation
    void iTransform() override;                                 // Inverse FFT transformation

private:
    // Private helper functions for FFT implementation
    void iterative_sequential(Mat& input_matrix, const unsigned int n);                              // Iterative sequential FFT
    void iterative_parallel(Mat& input_matrix, const unsigned int n, const unsigned int numThreads); // Iterative parallel FFT
};
