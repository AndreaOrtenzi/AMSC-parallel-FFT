#ifndef ARRAY_LENGTH
#define ARRAY_LENGTH 8
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

#define Real double

#include <iostream>
#include "GetPot"
#include <string.h>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <omp.h> 

#ifndef SEQ_IMPL
#define SEQ_IMPL true
#endif

#ifndef PAR_IMPL
#define PAR_IMPL true
#endif

// #include "../inc/AbstractFFT.hpp"
#if SEQ_IMPL
#include "../inc/SequentialFFT.hpp"
#endif

// #include <ctime>

// recursiveFFT function requires the input size n to be
// a power of 2. In contrast, the previous DFT implementation
// can handle input sizes of any length.

using namespace std;

void DFT(complex<Real> x[], const unsigned int n){
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

#if CHECK_CORRECTNESS
int checkCorrectness(const string implemName, const vector<complex<Real>> &correct, const vector<complex<Real>> &toCheck) {
    bool isCorrect = true;
    constexpr Real eps(1e-10*MAX_ARRAY_VALUES);

    auto j = toCheck.begin();
    for ( auto i = correct.begin(); i!=correct.end(); ++i) {
        if ( (i->imag()-j->imag()) > eps || (i->real()-j->real()) > eps) {
            std::cout << "Problem with element: " << *i << ", It should be: " << *j << endl;
            isCorrect = false;
        }
        if (j!=toCheck.end())
            j++;
    }
    if (!isCorrect) {
        std::cout << "WRONG TRANSFORMATION!" << endl;// \nFirst difference in "<< implem_name <<": x[" << i << "] = " << x[i] << ", y[" << i << "] = " << y << endl;
        return 1;
    }
    std::cout << "Correct transformation!" << endl;
    return 0;
}
#endif

void fillArray(vector<complex<Real>> &toFill, unsigned int seed = 10){
    srand(time(nullptr)*seed*0.1);
    for (std::vector<complex<Real>>::iterator it = toFill.begin(); it != toFill.end(); ++it){
        // complex<Real> temp = complex<Real>(rand() * 1.0/ RAND_MAX, rand() * 1.0/ RAND_MAX);
        *it = complex<Real>((int) (MAX_ARRAY_VALUES/ RAND_MAX * rand()),(int) (MAX_ARRAY_VALUES / RAND_MAX * rand()));

    }
        
}

int main(int argc, char *argv[]) {
    GetPot cmdLine(argc, argv);

    // Set testing parameters
    const unsigned int vectorLength = cmdLine.follow(ARRAY_LENGTH, "-N");
    const unsigned int iterToTime = cmdLine.follow(NUM_ITER_TO_TIME, "-iTT");
    
    

    // set all it's needed to time the execution
    #if TIME_IMPL
    using clock = std::chrono::steady_clock;
	using unitOfTime = std::chrono::duration<double, std::milli>;
    const string unitTimeStr = "ms";

    chrono::time_point<clock> begin;
    double total = 0.0;
    #endif
    
    
    // create the array to convert with FFT
    vector<complex<Real>> xSpace(vectorLength);
    fillArray(xSpace);
    vector<complex<Real>> xFreq(xSpace);
    
    #if CHECK_CORRECTNESS    
    // run the recursive version
    {
        unsigned int i = 0;
        #if TIME_IMPL
        for(i = 0; i < iterToTime; i++ ){            
        #endif
            // print out the result to check if the recursive version is correct
            std::cout << "Space values:" << endl;
            for (std::vector<complex<Real>>::iterator it = xSpace.begin(); it != xSpace.end(); ++it)
                std::cout << "\t" << *it;
            std::cout << endl;
        #if TIME_IMPL
            begin = clock::now();
            DFT(xSpace.data(),xSpace.size());
        #else
            DFT(xFreq.data(),xFreq.size());
        #endif

        

        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            total += elapsed;

            // print out the result to check if the recursive version is correct
            std::cout << "Frequency values:" << endl;
            for (std::vector<complex<Real>>::iterator it = xSpace.begin(); it != xSpace.end(); ++it)
                std::cout << "\t" << *it;
            std::cout << endl;
        #else
            // print out the result to check if the recursive version is correct
            std::cout << "Frequency values:" << endl;
            for (std::vector<complex<Real>>::iterator it = xFreq.begin(); it != xFreq.end(); ++it)
                std::cout << "\t" << *it;
            std::cout << endl;
        #endif
        #if TIME_IMPL
            // create a new test vector every iteration
            fillArray(xSpace,i+1);
        }    
        std::cout << "Recursive took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
    }    
    #endif

    // run my implementations:
    const vector<complex<Real>> empty_vec(vectorLength);

    // sequential implementation:  
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
            DFT(xSpace.data(),xSpace.size());
            #if CHECK_CORRECTNESS
                checkCorrectness(implementationName, fft.getFrequencyValues(), xSpace);
            #endif
        #else
        #if CHECK_CORRECTNESS
            checkCorrectness(implementationName, fft.getFrequencyValues(), xFreq);
        #endif
        #endif
        #if TIME_IMPL
            fillArray(xSpace,i+1);
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "--------------------------------\n" << endl;
    }
    #endif

    //parallel implementation: 
    #if PAR_IMPL
    {
        const string implementationName = "Parallel implementation";
        std::cout << "----------------"<< implementationName <<"----------------" << endl;

        unsigned int i = 0;
        #if TIME_IMPL
        total = 0.0;
        for( i = 0; i < iterToTime; i++ ){
        #endif
        
        ParallelFFT par_fft(xSpace, empty_vec);
        
        #if TIME_IMPL
            begin = clock::now();
        #endif

        par_fft.transform();
        
        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
            total += elapsed;
            DFT(xSpace.data(),xSpace.size());
            #if CHECK_CORRECTNESS
                checkCorrectness(implementationName, par_fft.getFrequencyValues(), xSpace);
            #endif
        #else
        #if CHECK_CORRECTNESS
            checkCorrectness(implementationName, par_fft.getFrequencyValues(), xFreq);
        #endif
        #endif
        #if TIME_IMPL
            fillArray(xSpace,i+1);
        }
        std::cout << implementationName << " took on average: " << total/iterToTime << unitTimeStr << endl;
        #endif
        std::cout << "--------------------------------\n" << endl;
    }
    #endif

    return 0;
}