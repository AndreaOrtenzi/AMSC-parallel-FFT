#include "FFT_2D.hpp"
#include "SequentialFFT.hpp"
#include "ParallelFFT.hpp"

#include <iostream>
#include <omp.h>
#include <cstdlib>
#include <cmath>
// #include <chrono>

using namespace std; 

void FFT_2D::generate_input(unsigned int pow)
{
    unsigned int size = std::pow(2, pow); // Calculate the size of the matrix

    input_matrix.resize(size, size); // Set input matrix as 2^pow x 2^pow matrix
    n = size; // Update private variable n

    // Generate random complex numbers between 0.0 and and 250.0 and fill the matrix
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            double real_part = (static_cast<double>(rand()) / RAND_MAX) * 250.0;
            double imag_part = (static_cast<double>(rand()) / RAND_MAX) * 250.0;
            input_matrix(i, j) = std::complex<double>(real_part, imag_part);
        }
    }
}

void FFT_2D::load_input(const std::string& filename) 
{
    SpMat mat_input;
    
    std::cout<<"Loading input from file:"<<std::endl;
    Eigen::loadMarket(mat_input, filename);

    // Check if the loaded matrix is square and has dimensions 2^pow x 2^pow
    unsigned int rows = mat_input.rows();
    unsigned int cols = mat_input.cols();
    unsigned int pow = static_cast<unsigned int>(log2(rows));

    if (rows != cols || (1U << pow) != rows) {
        std::cerr << "Error: The loaded matrix should be square and have dimensions 2^pow x 2^pow." << std::endl;
        return;
    }

    // Assign the loaded matrix to the input_matrix and update private variable n
    input_matrix = mat_input;
    n = rows;
    std::cout<<"Loading has been successful." <<std::endl;
}

// Using method from SequentialFFT class:
void FFT_2D::iterative_sequential() 
{
    // First pass: Apply FFT to each row
    for (unsigned int i = 0; i < n; ++i) {
        SpVec row_vector = input_matrix.row(i);
        SequentialFFT::iterativeFFT(row_vector.data(), n);
        iter_seq_sol.row(i) = row_vector;
    }

    // Second pass: Apply FFT to each column of the result from the first pass
    for (unsigned int j = 0; j < n; ++j) {
        SpVec column_vector = iter_seq_sol.col(j);
        SequentialFFT::iterativeFFT(column_vector.data(), n);
        iter_seq_sol.col(j) = column_vector;
    }
}

void FFT_2D::iterative_parallel()
{
    SpVec temp_vector;

}


