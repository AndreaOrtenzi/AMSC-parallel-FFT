#include "../inc/parameters"

#include <iostream>
#include <string.h>

#include <vector>
#include <complex>

#if TIME_IMPL
#include <chrono>
#endif

#if PAR_IMPL
#include <omp.h> 
#include "../inc/ParFFT2D.hpp"
#endif

#if SEQ_IMPL
#include "../inc/SeqFFT2D.hpp"
#endif

#include "../../lib/GetPot"

#if CHECK_CORRECTNESS

/**
 * Perform a 2D Discrete Fourier Transform (DFT) on the input spatialValues.
 * Store the resulting frequency domain values in the frequencyValues matrix.
 *
 * @param spatialValues     The input 2D matrix of spatial domain values.
 * @param frequencyValues   The output 2D matrix to store frequency domain values.
 */
template <class T>
void DFT2D(const std::vector<std::vector<T>> &spatialValues, std::vector<std::vector<std::complex<double>>> &frequencyValues) {
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

/**
 * Check the correctness of a Fourier transformation implementation by comparing
 * the results (frequency or spatial) in 'toCheck' with the correct values in 'correct'.
 *
 * @param implemName  A descriptive name of the implementation being tested.
 * @param correct     Reference to the matrix of correct values.
 * @param toCheck     Reference to the matrix of values to be checked.
 * @return            0 if the transformation is correct, 1 otherwise.
 */
template <class T>
int checkCorrectness(const std::string implemName, const std::vector<std::vector<T>> &correct, const std::vector<std::vector<T>> &toCheck) {
    bool isCorrect = true;
    constexpr double eps(1e-10 * MAX_MAT_VALUES);

    for (uint64_t i = 0; i < correct.size(); ++i) {
        for (uint64_t j = 0; j < correct[0].size(); ++j) {
            const std::complex<double> &correctValue = correct[i][j];
            const std::complex<double> &toCheckValue = toCheck[i][j];

            if (std::abs(correctValue.imag() - toCheckValue.imag()) > eps ||
                std::abs(correctValue.real() - toCheckValue.real()) > eps) {
                std::cout << "Problem with element at (" << i << ", " << j << "): " << toCheckValue
                          << ", It should be: " << correctValue << std::endl;
                isCorrect = false;
            }
        }
    }

    if (!isCorrect) {
        std::cout << "WRONG TRANSFORMATION in " << implemName << "!" << std::endl;
        return 1;
    }

    std::cout << "Correct transformation in " << implemName << "!" << std::endl;
    return 0;
}
#endif

/**
 * Fill a 2D matrix with random values.
 *
 * @param matToFill  Reference to the matrix to be filled with random values.
 * @param pow        The power of 2 that determines the size of the matrix (2^pow x 2^pow).
 * @param seed       Seed value for the random number generator (default is 10).
 */
template <class T>
void fill_input_matrix(std::vector<std::vector<T>> &matToFill, unsigned int pow, unsigned int seed = 10)
{
    srand(time(nullptr)*seed*0.1);
    unsigned int size = std::pow(2, pow); // Calculate the size of the matrix

    
    // Set input matrix as 2^pow x 2^pow matrix
    matToFill.resize(size);

    // Generate random complex numbers between 0.0 and and 250.0 and fill the matrix
    for (unsigned int i = 0; i < size; ++i) {
        matToFill[i].resize(size,0);
        for (unsigned int j = 0; j < size; ++j) {
            matToFill[i][j] = static_cast<T>( static_cast<double>(rand()* RAND_MAX) / MAX_MAT_VALUES);
        }
    }
}

unsigned int print_i = 0;
/**
 * Print a matrix of values.
 *
 * @param vec   The 2D vector to be printed.
 * @param name  An optional name for the printed vector (default is an incremental index).
 */
template <class T>
void printInt(const std::vector<std::vector<T>> &vec, const std::string &name = "") {
    // Determine the name to be used for printing
    std::string printName;
    if (name.empty()) {
        printName = "Vector" + std::to_string(print_i++);
    } else {
        printName = name;
    }
    std::cout << "Print " << printName << std::endl;

    if (vec.size() > 10) {
        std::cout << "Too many elements" << std::endl;
        return;
    }
    
    // Print the elements of the vector
    for (auto i : vec){
        for (auto j : i) {
            const std::complex<double> val = static_cast<std::complex<double>>(j);
            std::cout << "(" << ROUND_TO_ZERO(val.real()) << ", " << ROUND_TO_ZERO(val.imag()) << ") ";
        }
        std::cout << std::endl;
    }
}

// Function to check if is a number is a power of 2
bool isPowerOf2(unsigned int num) {
    return (num != 0) && ((num & (num - 1)) == 0);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    GetPot cmdLine(argc, argv);

    // Set testing parameters:
    const unsigned int rowlength = cmdLine.follow(ROW_LENGTH, "-N");
    const unsigned int iterToTime = cmdLine.follow(NUM_ITER_TO_TIME, "-iTT");
    const unsigned int numThreads = cmdLine.follow(NUM_THREADS == 0 ? \
        omp_get_max_threads() > rowlength ? rowlength : omp_get_max_threads() \
        : NUM_THREADS, "-nTH");
    
    // Check input row_length:
    if (!isPowerOf2(rowlength)) {
        std::cerr << "Error: row length must be a power of 2." << std::endl;
        return 1;
    }

    
    std::cout << "---------------- FFT2D on matrices "<< rowlength << "x" << rowlength <<" ----------------" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl << std::endl;
    
    // Set up timing
    #if TIME_IMPL
    using clock = std::chrono::steady_clock;
    using unitOfTime = std::chrono::duration<double, std::micro>;
    const std::string unitTimeStr = " \u03BCs";

    std::chrono::time_point<clock> begin;
    double total = 0.0;
    #endif

    // Create the matrix xSpace to convert with FFT
    const unsigned int pow = std::log2(rowlength);
    std::vector<std::vector<unsigned char>> xSpace, xSpaceToCheck;
    std::vector<std::vector<std::complex<double>>> xFreq, correctXFreq;
    fill_input_matrix(xSpace, pow);

    xFreq.resize(xSpace.size());
    for (unsigned int i = 0; i < xSpace.size(); i++)
        xFreq[i].reserve(xSpace[i].size());

    #if CHECK_CORRECTNESS
    // Run the O(n^2) version to check correctness
    {
        unsigned int i = 0;
        #if TIME_IMPL
        for (i = 0; i < iterToTime; i++) {
            printInt(xSpace, "space values DFT:");

            begin = clock::now();
        #endif
        
        // Perform a reference DFT transformation
        DFT2D(xSpace, xFreq);
        
        #if TIME_IMPL
            double elapsed = std::chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            total += elapsed;

            printInt(xFreq, "freq values DFT:");

            // Create a new test matrix every iteration
            fill_input_matrix(xSpace, pow, i + 1);
        }
        std::cout << "DFT2D took on average: " << total / iterToTime << unitTimeStr << std::endl;
        #endif
    }
    #endif

    // Run custom implementations:

    // Sequential implementation
    #if SEQ_IMPL
    {
        const std::string implementationName = "Sequential FFT 2D implementation";
        std::cout << "----------------" << implementationName << "----------------" << std::endl;

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for (i = 0; i < iterToTime; i++) {
            begin = clock::now();
        #endif

        // Perform the sequential FFT transformation
        SeqFFT2D::transform(xSpace, xFreq);
        
        #if TIME_IMPL
            double elapsed = std::chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr << " in the execution number " << i << std::endl;
            total += elapsed;
        #endif
        
        #if CHECK_CORRECTNESS
            // Compute the correct DFT for comparison
            DFT2D(xSpace, correctXFreq);
            checkCorrectness(implementationName, correctXFreq, xFreq);
            
            std::cout << "Check correctness of the iFFT: " << std::endl;

            // Perform inverse FFT for correctness check
            SeqFFT2D::iTransform(correctXFreq, xSpaceToCheck);
            checkCorrectness(implementationName + " inverse", xSpace, xSpaceToCheck);
        #endif

        #if TIME_IMPL
            fill_input_matrix(xSpace, pow, i + 1);
        }
        std::cout << implementationName << " took on average: " << total / iterToTime << unitTimeStr << std::endl;
        #endif
        std::cout << "----------------------------------------------------------------------------\n" << std::endl;
    }
    #endif

    // Parallel implementation
    #if PAR_IMPL
    {
        const std::string implementationName = "Parallel FFT 2D implementation with " + std::to_string(numThreads) + " threads";
        std::cout << "----------------" << implementationName << "----------------" << std::endl;

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for (i = 0; i < iterToTime; i++) {
            begin = clock::now();
        #endif

        // Perform the parallel FFT transformation
        ParFFT2D::transform(xSpace, xFreq, numThreads);
        
        #if TIME_IMPL
            double elapsed = std::chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr << " in the execution number " << i << std::endl;
            total += elapsed;
        #endif
        
        #if CHECK_CORRECTNESS
            // Compute the correct DFT for comparison
            DFT2D(xSpace, correctXFreq);
            checkCorrectness(implementationName, correctXFreq, xFreq);
            
            std::cout << "Check inverse correctness of the inverse: " << std::endl;

            // Perform inverse FFT for correctness check
            ParFFT2D::iTransform(correctXFreq, xSpaceToCheck, numThreads);
            checkCorrectness(implementationName + " inverse", xSpace, xSpaceToCheck);
        #endif

        #if TIME_IMPL
            fill_input_matrix(xSpace, pow, i + 1);
        }
        std::cout << implementationName << " took on average: " << total / iterToTime << unitTimeStr << std::endl;
        #endif
        std::cout << "----------------------------------------------------------------------------\n" << std::endl;
    }
    #endif

    return 0;
}


