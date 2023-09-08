#ifndef ARRAY_LENGTH
#define ARRAY_LENGTH 256
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
#ifndef MAX_ARRAY_VALUES
#define MAX_ARRAY_VALUES 250.0
#endif

#ifndef Real
#define Real double
#endif

#include <iostream>
#include "../../lib/GetPot"
#include <string.h>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <vector>

#ifndef SEQ_IMPL
#define SEQ_IMPL true
#endif
#ifndef PAR_OMP_IMPL
#define PAR_OMP_IMPL true
#endif
#ifndef PAR_MPI_IMPL
#define PAR_MPI_IMPL true
#endif

// #include "../inc/AbstractFFT.hpp"
#if SEQ_IMPL
#include "../inc/SequentialFFT.hpp"
#endif

#if PAR_OMP_IMPL
#include <omp.h> 
#include "../inc/Parallel_OMP_FFT.hpp"
#endif

#if PAR_MPI_IMPL
#include <mpi.h>
#include "../inc/Parallel_MPI_FFT.hpp"
#endif

// #include <ctime>

// recursiveFFT function requires the input size n to be
// a power of 2. In contrast, the previous DFT implementation
// can handle input sizes of any length.

using namespace std;

// Function to perform the Discrete Fourier Transform
void DFT(complex<Real> x[], const unsigned int n) {
    complex<Real> frequencyValues[n];

    for (unsigned int k = 0; k < n; ++k) {
        std::complex<Real> sum(0, 0);
        for (unsigned int j = 0; j < n; ++j) {
            std::complex<Real> term = x[j] * std::exp(-2.0 * M_PI * std::complex<Real>(0, 1) * static_cast<Real>(k * j) / static_cast<Real>(n));
            
            sum += term;
        }
        frequencyValues[k] = sum;
    }
    for (unsigned int k = 0; k < n; ++k) {
        x[k] = frequencyValues[k];
    }
}

// Function to check the correctness of FFT results:
#if CHECK_CORRECTNESS
int checkCorrectness(const string implemName, const vector<complex<Real>> &correct, const vector<complex<Real>> &toCheck) {
    bool isCorrect = true;
    constexpr Real eps(1e-10*MAX_ARRAY_VALUES);
    int pos = 0;

    auto j = toCheck.begin();
    for ( auto i = correct.begin(); i != correct.end(); ++i) {
        if ((i->imag() - j->imag()) > eps || (i->real() - j->real()) > eps) {
            std::cout << "Problem with element at index " << pos << ": " << *j << ", It should be: " << *i << endl;
            isCorrect = false;
        }
        pos++;
        if (j != toCheck.end())
            j++;
    }
    if (!isCorrect) {
        std::cout << "WRONG TRANSFORMATION!" << endl;
        return 1;
    }
    std::cout << "Correct transformation " << implemName << "!" << endl;
    return 0;
}
#endif

// Function to fill an array with random complex values:
void fillArray(vector<complex<Real>> &toFill, unsigned int seed = 10){
    srand(time(nullptr)*seed*0.1);
    for (std::vector<complex<Real>>::iterator it = toFill.begin(); it != toFill.end(); ++it){
        *it = complex<Real>((int) (MAX_ARRAY_VALUES/ RAND_MAX * rand()), (int) (MAX_ARRAY_VALUES / RAND_MAX * rand()));
    }
}

int main(int argc, char *argv[]) {
    GetPot cmdLine(argc, argv);

    // Set testing parameters based on command line arguments or defaults:
    const unsigned int vectorLength = cmdLine.follow(ARRAY_LENGTH, "-N");
    const unsigned int iterToTime = cmdLine.follow(NUM_ITER_TO_TIME, "-iTT");

    // Configure timing options:
    #if TIME_IMPL
    using clock = std::chrono::steady_clock;
    using unitOfTime = std::chrono::duration<double, std::milli>;
    const string unitTimeStr = "ms";

    chrono::time_point<clock> begin;
    double total = 0.0;
    #endif

    // Create the array to convert with FFT:
    vector<complex<Real>> xSpace(vectorLength);
    fillArray(xSpace);
    vector<complex<Real>> xFreq(xSpace);
    const vector<complex<Real>> empty_vec(vectorLength);

    // Initialize MPI environment if the parallel MPI implementation is enabled.
    #if PAR_MPI_IMPL
    int world_size = 1, world_rank = 0;

    MPI_Init(NULL, NULL); // Initialize MPI environment
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (world_rank == 0) {
    #endif

    #if CHECK_CORRECTNESS
    // Run the DFT for correctness comparison:
    {
        unsigned int i = 0;
        #if TIME_IMPL
        for(i = 0; i < iterToTime; i++ ){            
            begin = clock::now();
        #endif
        DFT(xFreq.data(), xFreq.size());

        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            total += elapsed;
        #endif
        
        #if TIME_IMPL
            // Create a new test vector every iteration:
            fillArray(xSpace,i+1);
            xFreq = xSpace;
        }    
        std::cout << "DFT took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
    }    
    #endif

    // Sequential implementation.
    #if SEQ_IMPL
    {
        const string implementationName = "Sequential implementation";
        std::cout << "----------------"<< implementationName <<"----------------" << endl;

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
        #endif
        
        SequentialFFT fft(xSpace, empty_vec);
        
        #if TIME_IMPL
            begin = clock::now();
        #endif

        fft.transform();
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
            DFT(xFreq.data(),xFreq.size());
        #endif
        
        #if CHECK_CORRECTNESS
            checkCorrectness(implementationName, xFreq, fft.getFrequencyValues());
            // Check the inverse:
            fft.iTransform();
            checkCorrectness(implementationName + " inverse", xSpace, fft.getSpatialValues());
        #endif
        
        #if TIME_IMPL
            fillArray(xSpace,i+1);
            xFreq = xSpace;
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "--------------------------------\n" << endl;
    }
    #endif // SEQ_IMPL

    // OpenMP implementation.
    #if PAR_OMP_IMPL
    {
        const string implementationName = "OpenMP implementation";
        std::cout << "----------------"<< implementationName <<"----------------" << endl;

        unsigned int i = 0;
        xFreq = xSpace;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
        #endif
        
        Parallel_OMP_FFT fft(xSpace, empty_vec);
        
        #if TIME_IMPL
            begin = clock::now();
        #endif

        fft.transform();
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
            DFT(xFreq.data(),xFreq.size());
        #endif
        
        #if CHECK_CORRECTNESS
            checkCorrectness(implementationName, xFreq, fft.getFrequencyValues());
            // Check the inverse:
            fft.iTransform();
            checkCorrectness(implementationName + " inverse", xSpace, fft.getSpatialValues());
        #endif
        
        #if TIME_IMPL
            fillArray(xSpace,i+1);
            xFreq = xSpace;
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "--------------------------------\n" << endl;
    }
    #endif // PAR_OMP_IMPL

    // MPI implementation.
    #if PAR_MPI_IMPL
    } // if (world_rank == 0) {
    {
        const string implementationName = "MPI implementation";
        if (world_rank == 0)
            std::cout << "----------------"<< implementationName <<"----------------" << endl;

        unsigned int i = 0;

        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
        #endif
        
        Parallel_MPI_FFT fft(xSpace, empty_vec, world_size, world_rank);
        MPI_Barrier(MPI_COMM_WORLD);

        #if TIME_IMPL
            begin = clock::now();
        #endif

        fft.transform();
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            if (world_rank==0)
                std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
        #endif
        
        #if CHECK_CORRECTNESS
            fft.iTransform();
            if (world_rank==0){
                DFT(xFreq.data(),xFreq.size());
                checkCorrectness(implementationName, xFreq, fft.getFrequencyValues());
                // Check the inverse:    
                checkCorrectness(implementationName + " inverse", xSpace, fft.getSpatialValues());
            }
        #endif
        
        #if TIME_IMPL
            if (world_rank==0)
                fillArray(xSpace,i+1);
            xFreq = xSpace;
        }
        if (world_rank==0)
            std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif

        MPI_Finalize(); // Finish MPI environment

        if (world_rank==0)
            std::cout << "--------------------------------\n" << endl;
    }
    #endif // PAR_MPI_IMPL

    return 0;
}