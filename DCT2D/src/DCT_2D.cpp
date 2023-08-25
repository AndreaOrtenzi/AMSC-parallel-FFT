#include "../inc/DCT_2D.hpp"

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <chrono>

using namespace std; 
using namespace std::chrono;

const Mat& DCT_2D::getSpatialValues() const {
    return spatialValues;
}

const Mat& DCT_2D::getFrequencyValues() const {
    return frequencyValues;
}


//The inverse of DCT-II is DCT-III multiplied by scaling factor 2/N (https://en.wikipedia.org/wiki/Discrete_cosine_transform)
void DCT_2D::inv_transform_1D(Vec& x) {
    unsigned int N = x.size();
    double scaling_factor = 2.0 / N; 

    for (unsigned int i = 0; i < N; i++) {
        double sum = x[0] / 2.0;
        for (unsigned int j = 1; j < N; j++) {
            double angle = M_PI / N * j * (i + 0.5);
            sum += x[j] * std::cos(angle);
        }
        x[i] = static_cast<int>(sum * scaling_factor);
    }
}


void DCT_2D::transform_par(){
    // Resize frequency matrix: 
    frequencyValues.resize(n, n);

    std::cout << "***Start Parallel Iterative Implementation***" << std::endl;
    frequencyValues = spatialValues;
    iterative_parallel(frequencyValues,frequencyValues.rows());
}

void DCT_2D::transform_seq(){
    // Resize frequency matrix: 
    frequencyValues.resize(n, n);

    std::cout << "***Start Sequential Iterative Implementation***" << std::endl;
    frequencyValues = spatialValues;
    iterative_sequential(frequencyValues,frequencyValues.rows());
}

// We have used the version II : DCT-II (https://en.wikipedia.org/wiki/Discrete_cosine_transform)
void DCT_2D::iterative_sequential(Mat& input_matrix, const unsigned int n) {
    
    // First pass: apply DCT to each row:
    for (unsigned int k = 0; k < n; k++) {
        Vec row_vector = input_matrix.row(k);
        for (unsigned int i = 0; i < n; i++) {
            double sum = 0.0;
            for (unsigned int j = 0; j < n; j++){
                sum += row_vector[j] * std::cos(M_PI / n * (j + 0.5) * i);
            }
            row_vector[i] = static_cast<int>(sum);
        }
        input_matrix.row(k) = row_vector;
    }

    // Second pass: apply DCT to each col:
    for (unsigned int k = 0; k < n; k++) {
        Vec col_vector = input_matrix.col(k);
        for (unsigned int i = 0; i < n; i++) {
            double sum = 0.0;
            for (unsigned int j = 0; j < n; j++){
                sum += col_vector[j] * std::cos(M_PI / n * (j + 0.5) * i);
            }
            col_vector[i] = static_cast<int>(sum);
        }
        input_matrix.col(k) = col_vector;
    }
}


void DCT_2D::iterative_parallel(Mat& input_matrix, const unsigned int n) {

    // Try with different number of threads:
    unsigned int numThreads = 8; 
    //unsigned int numThreads = static_cast<unsigned int>( log2(n) );
    // unsigned int numThreads = 2;
    // unsigned int numThreads = n;

    // First pass: Apply DCT to each row in parallel
    for (unsigned int k = 0; k < n; k++) {
        Vec row_vector = input_matrix.row(k);
        #pragma omp task shared(row_vector) firstprivate(n)
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for (unsigned int i = 0; i < n; i++) {
            double sum = 0.0;
            for (unsigned int j = 0; j < n; j++) {
                sum += row_vector[j] * std::cos(M_PI / n * (j + 0.5) * i);
            }
            row_vector[i] = static_cast<int>(sum);
        }
        input_matrix.row(k) = row_vector;
    }

    // Second pass: Apply DCT to each column in parallel
    for (unsigned int k = 0; k < n; k++) {
        Vec col_vector = input_matrix.row(k);
        #pragma omp task shared(col_vector) firstprivate(n)
        #pragma omp parallel for num_threads(numThreads) schedule(static)
        for (unsigned int i = 0; i < n; i++) {
            double sum = 0.0;
            for (unsigned int j = 0; j < n; j++) {
                sum += col_vector[j] * std::cos(M_PI / n * (j + 0.5) * i);
            }
            col_vector[i] = static_cast<int>(sum);
        }
        input_matrix.col(k) = col_vector;
    }
}

void DCT_2D::iTransform() {
    // Perform the inverse DCT 2D on the frequency values and store the result in the spatial values
    spatialValues.resize(n, n);

    // First pass: apply inverse DCT 1D on each row:
    for (unsigned int i = 0; i < n; ++i) {
        Vec row_vector = frequencyValues.row(i);
        DCT_2D::inv_transform_1D(row_vector);
        spatialValues.row(i) = row_vector;
    }

    // Second pass: apply inverse DCT 1D on each column:
    for (unsigned int i = 0; i < n; ++i) {
        Vec col_vector = spatialValues.col(i);
        DCT_2D::inv_transform_1D(col_vector);
        spatialValues.col(i) = col_vector;
    }
}





