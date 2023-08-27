#ifndef ROW_LENGTH
#define ROW_LENGTH 4
#endif
#ifndef CHECK_CORRECTNESS
#define CHECK_CORRECTNESS true
#endif
#ifndef TIME_IMPL
#define TIME_IMPL true
#endif
#if TIME_IMPL
#include <chrono>
#ifndef NUM_ITER_TO_TIME
#define NUM_ITER_TO_TIME 2
#endif
#else
#define NUM_ITER_TO_TIME 1
#endif
#ifndef MAX_MAT_VALUES
#define MAX_MAT_VALUES 250
#endif


#include <iostream>
#include <string.h>
#include <iomanip>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <omp.h> 
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra> 

#include "GetPot"
#include "../inc/FFT_2D.hpp"

#ifndef SEQ_IMPL
#define SEQ_IMPL true
#endif

#ifndef PAR_IMPL
#define PAR_IMPL true
#endif

using namespace std;
using SparseMat = Eigen::SparseMatrix<std::complex<double>>;

void DFT_2D(Mat& spatialValues, const unsigned int n) {
    Mat frequencyValues(n, n);

    for (unsigned int k = 0; k < n; k++) {
        for (unsigned int l = 0; l < n; l++) {
            std::complex<double> sum(0, 0);
            for (unsigned int j = 0; j < n; j++) {
                for (unsigned int i = 0; i < n; i++) {
                    std::complex<double> term = spatialValues.coeff(i, j) *
                        std::exp(-2.0 * M_PI * std::complex<double>(0, 1) * static_cast<double>((k * i + l * j)) / static_cast<double>(n));
                    
                    sum += term;
                }
            }
            frequencyValues(k, l) = sum;
        }
    }

    // Copy the frequency values back to the spatial values matrix:
    for (unsigned int k = 0; k < n; ++k) {
        for (unsigned int l = 0; l < n; ++l) {
            spatialValues(k, l) = frequencyValues(k, l);
        }
    }
}

template <class T>
void DFT_2D(const std::vector<std::vector<T>> &spatialValues, std::vector<std::vector<std::complex<double>>> &frequencyValues) {
    const unsigned int n = spatialValues.size();
    frequencyValues.resize(n);

    for (unsigned int k = 0; k < n; k++) {
        frequencyValues[k].resize(n);
        for (unsigned int l = 0; l < n; l++) {
            std::complex<double> sum(0, 0);
            for (unsigned int j = 0; j < n; j++) {
                for (unsigned int i = 0; i < n; i++) {
                    std::complex<double> term = static_cast<std::complex<double>>(spatialValues[i][j]) *
                        std::exp(-2.0 * M_PI * std::complex<double>(0, 1) * static_cast<double>((k * i + l * j)) / static_cast<double>(n));
                    
                    sum += term;
                }
            }
            frequencyValues[k][l] = sum;
        }
    }
}

#if CHECK_CORRECTNESS
int checkCorrectness(const string implemName, const Mat &correct, const Mat &toCheck) {
    bool isCorrect = true;
    constexpr double eps(1e-10 * MAX_MAT_VALUES);

    for (int i = 0; i < correct.rows(); ++i) {
        for (int j = 0; j < correct.cols(); ++j) {
            const std::complex<double> &correctValue = correct.coeff(i, j);
            const std::complex<double> &toCheckValue = toCheck.coeff(i, j);

            if (std::abs(correctValue.imag() - toCheckValue.imag()) > eps ||
                std::abs(correctValue.real() - toCheckValue.real()) > eps) {
                std::cout << "Problem with element at (" << i << ", " << j << "): " << toCheckValue
                          << ", It should be: " << correctValue << endl;
                isCorrect = false;
            }
        }
    }

    if (!isCorrect) {
        std::cout << "WRONG TRANSFORMATION in " << implemName << "!" << endl;
        return 1;
    }

    std::cout << "Correct transformation in " << implemName << "!" << endl;
    return 0;
}

template <class T>
int checkCorrectness(const string implemName, const Mat &correct, const std::vector<std::vector<T>> &toCheck) {
    bool isCorrect = true;
    constexpr double eps(1e-10 * MAX_MAT_VALUES);

    for (int i = 0; i < correct.rows(); ++i) {
        for (int j = 0; j < correct.cols(); ++j) {
            const std::complex<double> &correctValue = correct.coeff(i, j);
            const std::complex<double> &toCheckValue = toCheck[i][j];

            if (std::abs(correctValue.imag() - toCheckValue.imag()) > eps ||
                std::abs(correctValue.real() - toCheckValue.real()) > eps) {
                std::cout << "Problem with element at (" << i << ", " << j << "): " << toCheckValue
                          << ", It should be: " << correctValue << endl;
                isCorrect = false;
            }
        }
    }

    if (!isCorrect) {
        std::cout << "WRONG TRANSFORMATION in " << implemName << "!" << endl;
        return 1;
    }

    std::cout << "Correct transformation in " << implemName << "!" << endl;
    return 0;
}

#endif

void fill_input_matrix(Mat& matToFill, unsigned int pow, unsigned int seed = 10)
{
    srand(time(nullptr)*seed*0.1);
    unsigned int size = std::pow(2, pow); // Calculate the size of the matrix

    matToFill.resize(size, size); // Set input matrix as 2^pow x 2^pow matrix

    // Generate random complex numbers between 0.0 and and 250.0 and fill the matrix
    for (unsigned int i = 0; i < size; ++i) {
        for (unsigned int j = 0; j < size; ++j) {
            double real_part = (static_cast<double>(rand()) / RAND_MAX) * MAX_MAT_VALUES;
            double imag_part = (static_cast<double>(rand()) / RAND_MAX) * MAX_MAT_VALUES;
            matToFill(i, j) = std::complex<double>(real_part, imag_part);
        }
    }
}

template <class T>
void fill_input_matrix(std::vector<std::vector<T>> &matToFill, unsigned int pow, unsigned int seed = 10)
{
    srand(time(nullptr)*seed*0.1);
    unsigned int size = std::pow(2, pow); // Calculate the size of the matrix

    
    // Set input matrix as 2^pow x 2^pow matrix
    matToFill.resize(size);

    // Generate random complex numbers between 0.0 and and 250.0 and fill the matrix
    for (unsigned int i = 0; i < size; ++i) {
        matToFill[i].resize(size);
        for (unsigned int j = 0; j < size; ++j) {
            matToFill[i][j] = static_cast<T>((rand() % MAX_MAT_VALUES)  ); 
        }
    }
}

void load_input(Mat& input_matrix, std::string& filename) 
{
    SparseMat sparse_mat_input;
    
    std::cout<<"Loading input from file:"<<std::endl;
    Eigen::loadMarket(sparse_mat_input, filename);

    // Check if the loaded matrix is square and has dimensions 2^pow x 2^pow
    unsigned int rows = sparse_mat_input.rows();
    unsigned int cols = sparse_mat_input.cols();
    unsigned int pow = static_cast<unsigned int>(log2(rows));

    // Error if #rows is different by #columns or their number are not power of 2:
    if (rows != cols || (1U << pow) != rows) {
        std::cerr << "Error: The loaded matrix should be square and have dimensions 2^pow x 2^pow." << std::endl;
        return;
    }

    // Assign the loaded matrix to the input_matrix and resize input_matrix
    input_matrix.resize(rows, cols);
    
    //Convert the sparse matrix in a dense format:
    for(unsigned int i=0; i<rows; i++){
        for(unsigned int j=0; j<cols; j++){
            input_matrix(i,j) = sparse_mat_input.coeff(i,j);
        }
    }
    std::cout<<"Loading has been successful." <<std::endl;
}



int main(int argc, char *argv[]) {
    GetPot cmdLine(argc, argv);

    // Set testing parameters:
    const unsigned int rowlength = cmdLine.follow(ROW_LENGTH, "-N");
    const unsigned int iterToTime = cmdLine.follow(NUM_ITER_TO_TIME, "-iTT");
    
    // set all it's needed to time the execution
    #if TIME_IMPL
    using clock = std::chrono::steady_clock;
	using unitOfTime = std::chrono::duration<double, std::milli>;
    const string unitTimeStr = "ms";

    chrono::time_point<clock> begin;
    double total = 0.0;
    #endif

    // create the matrix xSpace to convert with FFT:
    Mat xSpace(rowlength, rowlength); 
    const unsigned int pow = std::log2(ROW_LENGTH);
    // Fill the matrix xSpace with random complex values:
    fill_input_matrix(xSpace, pow);
    
    std::vector<std::vector<unsigned char>> vecXSpace;
    std::vector<std::vector<std::complex<double>>> vecXFreq;
    fill_input_matrix(vecXSpace, pow);

    Mat xFreq(xSpace);

    #if CHECK_CORRECTNESS    
    // run the recursive version
    {
        unsigned int i = 0;
        #if TIME_IMPL
        for(i = 0; i < iterToTime; i++ ){            
        #endif
            
        std::cout << "Space values:" << std::endl;
        for (int j = 0; j < xSpace.rows(); ++j) 
        {
            for (int k = 0; k < xSpace.cols(); ++k) {
                std::complex<double> value = xSpace(j, k);
                std::cout << "\t(" << std::fixed << std::setprecision(3) << value.real() << ", " << value.imag() << ")";
            }       
        std::cout << std::endl;
        }
        #if TIME_IMPL
            begin = clock::now();
            DFT_2D(xSpace,xSpace.rows());
        #else
            DFT_2D(xFreq.data(),xFreq.size());
        #endif
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            total += elapsed;

            
            std::cout << "Frequency values:" << endl;
            for (int j = 0; j < xSpace.rows(); ++j) 
            {
                for (int k = 0; k < xSpace.cols(); ++k) {
                std::complex<double> value = xSpace(j, k);
                std::cout << "\t(" << std::fixed << std::setprecision(3) << value.real() << ", " << value.imag() << ")";
                }       
            std::cout << std::endl;
            }
        #else
            // print out the result to check if the recursive version is correct
            std::cout << "Frequency values:" << endl;
            for (std::vector<complex<double>>::iterator it = xFreq.begin(); it != xFreq.end(); ++it)
                std::cout << "\t" << *it;
            std::cout << endl;
        #endif
        #if TIME_IMPL
            // create a new test matrix every iteration
            fill_input_matrix(xSpace, pow, i+1);
        }    
        std::cout << "DFT2D took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
    }    
    #endif

    // run my implementations:
    Mat empty_matrix(rowlength, rowlength);

    // sequential implementation:  
    #if SEQ_IMPL
    {
        const string implementationName = "Sequential FFT 2D implementation";
        std::cout << "----------------"<< implementationName <<"----------------" << endl;

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
        #endif
        
        FFT_2D fft2D(xSpace, empty_matrix);

        std::cout<<"Frequency values:" <<std::endl;
        for (int j=0; j<xSpace.rows(); j++){
            for(int k=0; k<xSpace.rows(); k++){
                xSpace(j, k) = vecXSpace[j][k];
            }
        }
        
        #if TIME_IMPL
            begin = clock::now();
        #endif

        fft2D.iterative_sequential(vecXSpace,vecXFreq);
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
            DFT_2D(xSpace,xSpace.rows());
            #if CHECK_CORRECTNESS
                checkCorrectness(implementationName, xSpace, vecXFreq);
            #endif
        #else
        #if CHECK_CORRECTNESS
            checkCorrectness(implementationName, fft2D.getFrequencyValues(), xFreq);
        #endif
        #endif
        #if TIME_IMPL
            fill_input_matrix(vecXSpace, pow, i+1);
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "----------------------------------------------------------------------------\n" << endl;
    }
    #endif

    Mat empty_mat(rowlength, rowlength);
    //parallel implementation: 
    #if PAR_IMPL
    {
        const string implementationName = "Parallel FFT 2D implementation";
        std::cout << "----------------"<< implementationName <<"----------------" << endl;

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
        #endif
        
        FFT_2D fft2D(xSpace, empty_mat);
        
        #if TIME_IMPL
            begin = clock::now();
        #endif

        fft2D.transform_par();
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
            DFT_2D(xSpace,xSpace.rows());
            #if CHECK_CORRECTNESS
                checkCorrectness(implementationName, fft2D.getFrequencyValues(), xSpace);
            #endif
        #else
        #if CHECK_CORRECTNESS
            checkCorrectness(implementationName, fft2D.getFrequencyValues(), xFreq);
        #endif
        #endif
        #if TIME_IMPL
            fill_input_matrix(xSpace, pow, i+1);
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "----------------------------------------------------------------------------\n" << endl;
    }
    #endif

    return 0;
}

