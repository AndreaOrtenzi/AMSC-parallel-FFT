#include "../inc/parameters"

#include <iostream>
#include <string.h>
// #include <iomanip>
// #include <cmath>
// #include <cstdlib>

#include <vector>
#include <complex>

#if TIME_IMPL
#include <chrono>
#endif

#if PAR_IMPL
#include <omp.h> 
#endif

#if SEQ_IMPL
#endif

#include "../inc/FFT_2D.hpp"
#include "../../lib/GetPot"


using namespace std;
using SparseMat = Eigen::SparseMatrix<std::complex<double>>;

#if CHECK_CORRECTNESS
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
    const unsigned int numThreads = cmdLine.follow(NUM_THREADS == 0 ? omp_get_max_threads() : NUM_THREADS, "-nTH");

    std::cout << "---------------- FFT2D on matrices "<< rowlength << "x" << rowlength <<" ----------------" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl << std::endl;
    
    // set all it's needed to time the execution
    #if TIME_IMPL
    using clock = std::chrono::steady_clock;
	using unitOfTime = std::chrono::duration<double, std::micro>;
    const string unitTimeStr = " \u03BCs";

    chrono::time_point<clock> begin;
    double total = 0.0;
    #endif

    // create the matrix xSpace to convert with FFT:
    const unsigned int pow = std::log2(ROW_LENGTH);
    Mat xSpace, xSpaceToCheck;
    Mat xFreq, correctXFreq;
    // Fill the matrix xSpace with random complex values:
    fill_input_matrix(xSpace, pow);
    
    xFreq.resize(xSpace.rows(), xSpace.cols());

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

        FFT_2D fft2D(xSpace, xFreq);

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
            begin = clock::now();
        #endif
        
        fft2D.transform_seq();
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
        #endif

        #if CHECK_CORRECTNESS
            DFT_2D(xSpace,correctXFreq);
            checkCorrectness(implementationName, correctXFreq, xFreq);

            std::cout << "Check correctness of the iFFT: " << std::endl;

            fft2D.iTransform();
            checkCorrectness("iFFT", fft2D.getSpatialValues(), xSpaceToCheck);
        #endif

        #if TIME_IMPL
            fill_input_matrix(xSpace, pow, i+1);
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "----------------------------------------------------------------------------\n" << endl;
    }
    #endif

    //parallel implementation: 
    #if PAR_IMPL
    {
        const string implementationName = "Parallel FFT 2D implementation with " + std::str(numThreads) + " threads";
        std::cout << "----------------"<< implementationName <<"----------------" << endl;

        FFT_2D fft2D(xSpace, xFreq);

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
            begin = clock::now();
        #endif

        fft2D.transform_par(numThreads);
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
        #endif

        #if CHECK_CORRECTNESS
            DFT_2D(xSpace,correctXFreq);
            checkCorrectness(implementationName, correctXFreq, xFreq);
            
            std::cout << "Check correctness of the inverse: " << std::endl;
            fft2D.iTransform();
            checkCorrectness("iFFT", fft2D.getSpatialValues(), xSpaceToCheck);
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

