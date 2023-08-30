#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <mpi.h>
// #include "ourparallelFFT/*"
#include "Parallel_MPI_FFT.hpp"

#define MAX_ARRAY_VALUES 255
#define ARRAY_SIZE 256

void fillArray(std::vector<std::complex<Real>> &toFill, unsigned int seed = 10){
    srand(time(nullptr)*seed*0.1);
    for (std::vector<std::complex<Real>>::iterator it = toFill.begin(); it != toFill.end(); ++it){
        // complex<Real> temp = complex<Real>(rand() * 1.0/ RAND_MAX, rand() * 1.0/ RAND_MAX);
        *it = std::complex<Real>((int) (MAX_ARRAY_VALUES/ RAND_MAX * rand()),(int) (MAX_ARRAY_VALUES / RAND_MAX * rand()));

    }
        
}

void DFT(std::complex<Real> x[], const unsigned int n){
    std::complex<Real> frequencyValues[n];

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

int checkCorrectness(const std::string implemName, const std::vector<std::complex<Real>> &correct, const std::vector<std::complex<Real>> &toCheck) {
    bool isCorrect = true;
    constexpr Real eps(1e-10*MAX_ARRAY_VALUES);
    int pos = 0;

    auto j = toCheck.begin();
    for ( auto i = correct.begin(); i!=correct.end(); ++i) {
        if ( (i->imag()-j->imag()) > eps || (i->real()-j->real()) > eps) {
            std::cout << "Problem with element at index " << pos << ": " << *j << ", It should be: " << *i << std::endl;
            isCorrect = false;
        }
        pos++;
        if (j!=toCheck.end())
            j++;
    }
    if (!isCorrect) {
        std::cout << "WRONG TRANSFORMATION!" << std::endl;// \nFirst difference in "<< implem_name <<": x[" << i << "] = " << x[i] << ", y[" << i << "] = " << y << endl;
        return 1;
    }
    std::cout << "Correct transformation!" << std::endl;
    return 0;
}



int main(int argc, char *argv[]) {
    
    MPI_Init(NULL, NULL);      // initialize MPI environment

    // create the array to convert with FFT
    std::vector<std::complex<Real>> xSpace(ARRAY_SIZE);
    fillArray(xSpace);
    std::vector<std::complex<Real>> xFreq(xSpace);
    const std::vector<std::complex<Real>> empty_vec(ARRAY_SIZE);
    
    
    int world_size = 1, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    {
        Parallel_MPI_FFT fft(xSpace, xFreq, world_size, world_rank);
        MPI_Barrier(MPI_COMM_WORLD);

        fft.transform();

        if (world_rank==0){
            DFT(xFreq.data(),xFreq.size());
            checkCorrectness("MPI implementation", xFreq, fft.getFrequencyValues());
        }
    } // this is important to destroy fft before MPI_Finalize

    MPI_Finalize();

    return 0;
}