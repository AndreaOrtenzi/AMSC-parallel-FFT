#ifndef PARALLEL_MPI_FFT_HPP
#define PARALLEL_MPI_FFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <mpi.h>
#include <iostream>

#include "AbstractFFT.hpp"

class Parallel_MPI_FFT : public AbstractFFT {
public:
    // Use AbstractFFT overloaded methods, overloading hides the parent's methods
    using AbstractFFT::transform;
    using AbstractFFT::iTransform;

    // Constructor 1: Initializes the Parallel_MPI_FFT object with a specified problem size,
    // the number of MPI processes (world_size), and the MPI rank of the current process (world_rank).
    Parallel_MPI_FFT(const unsigned int problemSize, int ws, int wr) :
         AbstractFFT(problemSize)
        , world_size(worldSizeIni(ws, N, wr))
        , world_rank(wr) {
            int color = wr < world_size ? 0 : MPI_UNDEFINED;
            MPI_Comm_split(MPI_COMM_WORLD, color, wr, &world_size_comm);
        }

    // Constructor 2: Initializes the Parallel_MPI_FFT object with spatial and frequency values,
    // the number of MPI processes (world_size), and the MPI rank of the current process (world_rank).
    Parallel_MPI_FFT(const std::vector<std::complex<real>>& sValues, const std::vector<std::complex<real>>& fValues, int ws, int wr) :
         AbstractFFT(sValues, fValues)
        , world_size(worldSizeIni(ws, N, wr)) 
        , world_rank(wr){
            int color = wr < world_size ? 0 : MPI_UNDEFINED;
            MPI_Comm_split(MPI_COMM_WORLD, color, wr, &world_size_comm);
        }

    void transform(const std::vector<std::complex<real>>& sValues) override;
    void iTransform(const std::vector<std::complex<real>>& fValues) override;

    // Destructor: Frees the MPI communicator if the rank is less than the world size.
    ~Parallel_MPI_FFT() {
        if (world_rank < world_size)
            MPI_Comm_free(&world_size_comm);
    };

protected:
    void iterativeFFT(std::complex<real> x[], const unsigned int n) override;
    void recursiveFFT(std::complex<real> x[], const unsigned int n) override;

private:
    // Helper function to determine the world size based on the input value, problem size, and rank.
    static int worldSizeIni(int ws_value, const unsigned int N, const int wr){
        // Use a power of 2 ranks:
        const unsigned int pwr = floor(log2(ws_value));
        n_splitting = N >> pwr;

        if (1 << pwr != ws_value && wr == 0)
            std::cout << "Rank number is not a power of two. It'll use only " << (1 << pwr) << " processes." << std::endl;
        return 1 << pwr;
    }

    // Number of MPI processes:
    const int world_size; 
    // Rank of the current MPI process:
    const int world_rank; 
    static unsigned int n_splitting;
    static bool isRecursive;
    MPI_Comm world_size_comm;
};

#endif // PARALLEL_MPI_FFT_HPP
