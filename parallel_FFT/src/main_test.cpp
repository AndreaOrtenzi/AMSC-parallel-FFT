#ifndef ARRAY_LENGTH
#define ARRAY_LENGTH 5
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
#endif
#define Real double

#include <iostream>
#include "GetPot"
#include <string.h>
#include <complex>
#include <cmath>
#include <cstdlib>
#include <vector>

// #include <ctime>

// #if CHECK_CORRECTNESS
// #include "../../inc/AbstractFFT.hpp"
// #include "FFTwReference.cpp"
// #endif

using namespace std;

void recursiveFFT(complex<Real> x[], const unsigned int n) {
    if (n <= 1) {
        return;
    }
    complex<Real> even[n/2], odd[n/2];
    for (unsigned int i = 0; i < n/2; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }
    recursiveFFT(even, n/2);
    recursiveFFT(odd, n/2);
    for (unsigned int i = 0; i < n/2; i++) {
        complex<Real> t((complex<Real>)std::polar(1.0, -2*M_PI*i/n) * odd[i]);
        x[i] = even[i] + t;
        x[i+n/2] = even[i] - t;
    }
}

#if CHECK_CORRECTNESS
int checkCorrectness(const string implemName,const vector<complex<Real>> &correct,const vector<complex<Real>> &toCheck){
    
    if (correct == toCheck){
        std::cout << implemName <<" is correct."<< endl;
        return 0;
    }
    std::cout << "WRONG TRANSFORMATION!"<<endl;// \nFirst difference in "<< implem_name <<": x[" << i << "] = " << x[i] << ", y[" << i << "] = " << y << endl;
    return 1;
    
}
#endif

void fillArray(vector<complex<Real>> &toFill, unsigned int seed = 10){
    srand(time(nullptr)*seed*0.1);
    for (std::vector<complex<Real>>::iterator it = toFill.begin(); it != toFill.end(); ++it){
        // complex<Real> temp = complex<Real>(rand() * 1.0/ RAND_MAX, rand() * 1.0/ RAND_MAX);
        *it = complex<Real>(255.0/ RAND_MAX * rand(), 255.0 / RAND_MAX * rand());

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

    #if CHECK_CORRECTNESS
    vector<complex<Real>> xFreq;
    xFreq=xSpace;
    #endif
    
    #if CHECK_CORRECTNESS    
    // run the recursive version
    {
        unsigned int i = 0;
        #if TIME_IMPL
        for(i = 0; i < NUM_ITER_TO_TIME; i++ ){
            // print out the result to check if the recursive version is correct
            std::cout << "Space values:" << endl;
            for (std::vector<complex<Real>>::iterator it = xFreq.begin(); it != xFreq.end(); ++it)
                std::cout << "\t" << *it;
            std::cout << endl;

            begin = clock::now();
        #endif

        recursiveFFT(xFreq.data(),xFreq.size());

        #if TIME_IMPL
            double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
            total += elapsed;

            // print out the result to check if the recursive version is correct
            std::cout << "Frequency values:" << endl;
            for (std::vector<complex<Real>>::iterator it = xFreq.begin(); it != xFreq.end(); ++it)
                std::cout << "\t" << *it;
            std::cout << endl;

            // create a new test vector every iteration
            fillArray(xFreq,i+1);
        }    
        std::cout << "Recursive took on average: " << total/NUM_ITER_TO_TIME << unitTimeStr << endl;
        #endif
    }    
    #endif

    // run my implementations:
    // complex<double> xt[ARR_SIZE];

    // // sequential implementation:  
    // {
    //     const string implementationName = "Sequential implementation";
    //     std::cout << "----------------"<< implementationName <<"----------------" << endl;

    //     #if TIME_IMPL
    //     total = 0.0;
    //     for( int i = 0; i < NUM_ITER_TO_TIME; i++ ){
    //         begin = clock::now();
    //     #endif

    //     fft(x,xt, ARR_SIZE);
        
    //     #if TIME_IMPL
    //         double elapsed = chrono::duration_cast<unitOfTime>(clock::now() - begin).count();
    //         std::cout << "It took " << elapsed << unitTimeStr <<" in the execution number "<< i << endl;
    //         total += elapsed;
    //     }
    //     std::cout << implementationName << " took on average: " << total/NUM_ITER_TO_TIME << unitTimeStr << endl;
    //     #endif
        
    //     #if CHECK_CORRECTNESS
    //         checkCorrectness(implementationName,reference.getFrequencyValues(),xt);
    //     #endif
    //     std::cout << "--------------------------------\n" << endl;
    // }

    return 0;
}