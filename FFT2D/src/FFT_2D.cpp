#include "../inc/FFT_2D.hpp"

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Getters for spatial and frequency matrices
const Mat& FFT_2D::getSpatialValues() const {
    return spatialValues;
}

const Mat& FFT_2D::getFrequencyValues() const {
    return frequencyValues;
}

// Parallel FFT transformation:
void FFT_2D::transform_par(const unsigned int numThreads){
    // Resize frequency matrix:
    frequencyValues.resize(n, n);

    // Copy spatial values to frequency values
    frequencyValues = spatialValues;

    // Perform parallel iterative FFT transformation
    iterative_parallel(frequencyValues, frequencyValues.rows(), numThreads);
}

// Sequential FFT transformation:
void FFT_2D::transform_seq(){
    // Resize frequency matrix:
    frequencyValues.resize(n, n);

    // Copy spatial values to frequency values
    frequencyValues = spatialValues;

    // Perform sequential iterative FFT transformation
    iterative_sequential(frequencyValues, frequencyValues.rows());
}

// Perform iterative FFT transformation sequentially
void FFT_2D::iterative_sequential(Mat& input_matrix, const unsigned int n){
    // Calculate the number of bits needed for FFT
    unsigned int numBits = static_cast<unsigned int>(log2(n));

    // First pass: Apply FFT to each row
    for (unsigned int i = 0; i < n; ++i) {
        Vec row_vector = input_matrix.row(i);

        // Bit-reversal
        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(row_vector[l], row_vector[j]);
            }
        }

        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));

            for (unsigned int k = 0; k < n; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        input_matrix.row(i) = row_vector;
    }

    // Second pass: Apply FFT to each column
    for (unsigned int i = 0; i < n; ++i) {
        Vec col_vector = input_matrix.col(i);

        // Bit-reversal
        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));

            for (unsigned int k = 0; k < n; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        input_matrix.col(i) = col_vector;
    }
}

// Perform iterative FFT transformation in parallel with OpenMP
void FFT_2D::iterative_parallel(Mat& input_matrix, const unsigned int n, const unsigned int numThreads){
    unsigned int numBits = static_cast<unsigned int>(log2(n));

    // First pass: Apply parallel FFT to each row
    #pragma omp parallel for num_threads(numThreads)
    for(unsigned int i = 0; i < n; i++){
        Vec row_vector = input_matrix.row(i);

        // Bit-reversal
        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(row_vector[l], row_vector[j]);
            }
        }

        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));

            for (unsigned int k = 0; k < n; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        input_matrix.row(i) = row_vector;
    }

    // Second pass: Apply parallel FFT to each column
    #pragma omp parallel for num_threads(numThreads)
    for(unsigned int i = 0; i < n; i++){
        Vec col_vector = input_matrix.col(i);

        // Bit-reversal
        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));

            for (unsigned int k = 0; k < n; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        input_matrix.col(i) = col_vector;
    }
}

// Inverse FFT transformation:
void FFT_2D::iTransform() {
    // Resize spatial matrix to match the size of the frequency matrix
    spatialValues.resize(n, n);
    
    // inverse FFT normalization factor 1/N:
    double N_inv = 1.0 / static_cast<double>(n);
    std::cout << "***Start Inverse FFT Implementation***" << std::endl;
    
    // Calculate the number of bits
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    
    // First pass: Apply inverse FFT to each row
    for (unsigned int i = 0; i < n; ++i) {
        Vec row_vector = frequencyValues.row(i);

        // Bit-reversal
        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(row_vector[l], row_vector[j]);
            }
        }

        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m)); // Inverse twiddle factor

            for (unsigned int k = 0; k < n; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        spatialValues.row(i) = row_vector;
    }

    // Second pass: Apply inverse FFT to each column
    for (unsigned int i = 0; i < n; ++i) {
        Vec col_vector = spatialValues.col(i);

        // Bit-reversal
        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s <= numBits; s++) {
            unsigned int m = 1U << s;
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m)); // Inverse twiddle factor

            for (unsigned int k = 0; k < n; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }

        spatialValues.col(i) = col_vector;
    }

    // Scale the spatial matrix by 1/N^2
    for (unsigned int i = 0; i < spatialValues.rows(); ++i) {
        for (unsigned int j = 0; j < spatialValues.cols(); ++j) {
            spatialValues(i, j) *= N_inv * N_inv;
        }
    }
}
