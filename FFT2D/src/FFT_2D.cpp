#include "../inc/FFT_2D.hpp"

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <chrono>

//template void FFT_2D::iterative_sequential(std::vector<std::vector<unsigned char>>& input_matrix, std::vector<std::vector<std::complex<double>>>& freq_matrix);

using namespace std; 
using namespace std::chrono;

const Mat& FFT_2D::getSpatialValues() const {
    return spatialValues;
}

const Mat& FFT_2D::getFrequencyValues() const {
    return frequencyValues;
}

//Same code for SequentialFFT::iTransform(const std::vector<std::complex<real>>& fValues), but I need the version 
// with Eigen vector for FFT2D::inverse_transform():
void FFT_2D::inv_transform_1D(Vec& x) {
    
    unsigned int n = x.size();
    unsigned int numBits = static_cast<unsigned int>(log2(n));

        for (unsigned int l = 0; l < n; l++) {
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(x[l], x[j]);
            }
        }

    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
        for (unsigned int k = 0; k < n; k += m) {
            std::complex<double> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<double> t = w * x[k + j + m / 2];
                std::complex<double> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    // Real coefficient 1/N :
    double N_inv = 1.0 / static_cast<double>(n);

    for (unsigned int i = 0; i < n; i++) {
        x[i] *=  N_inv;
    }
}

void FFT_2D::transform_par(const unsigned int numThreads){
    // Resize frequency matrix: 
    frequencyValues.resize(n, n);

    std::cout << "***Start Parallel Iterative Implementation***" << std::endl;
    frequencyValues = spatialValues;
    iterative_parallel(frequencyValues,frequencyValues.rows(), numThreads);
}

void FFT_2D::transform_seq(){
    // Resize frequency matrix: 
    frequencyValues.resize(n, n);

    std::cout << "***Start Sequential Iterative Implementation***" << std::endl;
    frequencyValues = spatialValues;
    iterative_sequential(frequencyValues,frequencyValues.rows());
}

void FFT_2D::iterative_sequential(Mat& input_matrix, const unsigned int n){
    
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    //First pass: Apply FFT to each row:
    for (unsigned int i = 0; i < n; ++i) {
        Vec row_vector = input_matrix.row(i);
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

    //Second pass: Apply FFT to each column
    for (unsigned int i = 0; i < n; ++i) {
        Vec col_vector = input_matrix.col(i);
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

void FFT_2D::iterative_parallel(Mat& input_matrix, const unsigned int n, const unsigned int numThreads){
   
   unsigned int numBits = static_cast<unsigned int>(log2(n));
    // Maximum number of threads
    // int numThreads = omp_get_max_threads();
    //******************************************************************
    //          Try with different numbers of threads 
    //unsigned int numThreads = static_cast<unsigned int>(ceil(log2(n)));
    //unsigned int numThreads = 2;
    // unsigned int numThreads = 4;
    // unsigned int numThreads = n;
    // ******************************************************************

// First pass: Let's compute the parallel iterative FFT on rows:
    for(unsigned int i=0; i<n; i++){
        Vec row_vector = input_matrix.row(i);
        // Create region of parallel tasks in order to do bit reverse for input vector x, n is shared among all the threads of the region:
        #pragma omp task shared(row_vector) firstprivate(n)
        omp_set_num_threads(numThreads);
        #pragma omp parallel for schedule(static)
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
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m)); // Twiddle factor
        
        #pragma omp parallel for schedule(static)
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
    

    // Second pass: let's compute the parallel iterative FFT on columns:
    for(unsigned int i=0; i<n; i++){
        Vec col_vector = input_matrix.col(i);
        // Create region of parallel tasks in order to do bit reverse for input vector x, n is shared among all the threads of the region:
        #pragma omp task shared(col_vector) firstprivate(n)
        omp_set_num_threads(numThreads);
        #pragma omp parallel for schedule(static)
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
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m)); // Twiddle factor
        
        #pragma omp parallel for schedule(static)
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

void FFT_2D::iTransform() {
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values:
    spatialValues.resize(n, n);
    // Real coefficient 1/N
    double N_inv = 1.0 / static_cast<double>(n);
    std::cout << "***Start Inverse FFT Implementation***" << std::endl;
    
    //First pass: apply inverse FFT1D on each row:
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    for (unsigned int i = 0; i < n; ++i) {
        Vec row_vector = frequencyValues.row(i);
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
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
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

    // Second pass: apply inverse FFT on each column:
    for (unsigned int i = 0; i < n; ++i) {
        Vec col_vector = spatialValues.col(i);
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
            std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
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
    // Factorize per 1/N^2:
    for (unsigned int i = 0; i < spatialValues.rows(); ++i){
        for(unsigned int j = 0; j < spatialValues.cols(); ++j){
            spatialValues(i, j) *= N_inv * N_inv;
        }
    }
}
