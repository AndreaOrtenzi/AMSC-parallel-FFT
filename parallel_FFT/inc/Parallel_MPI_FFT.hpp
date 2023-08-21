#ifndef PARALLEL_MPI_FFT_HPP
#define PARALLEL_MPI_FFT_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <mpi.h>

#include "AbstractFFT.hpp"

class Parallel_MPI_FFT : public AbstractFFT {
public:
    // use AbstractFFT overloaded methods, overloading hides the parent's methods
    using AbstractFFT::transform;
    using AbstractFFT::iTransform;

    Parallel_MPI_FFT(const unsigned int problemSize, int ws, int wr) :
         AbstractFFT(problemSize)
        , world_rank(wr)
        , world_size(worldSizeIni(ws, N, wr)){
            int color = wr < world_size ? 0 : MPI_UNDEFINED;
            MPI_Comm_split(MPI_COMM_WORLD, color, wr, &world_size_comm);
        }

    Parallel_MPI_FFT(const std::vector<std::complex<real>>& sValues,const std::vector<std::complex<real>>& fValues, int ws, int wr) :
         AbstractFFT(sValues, fValues) 
        , world_rank(wr)
        , world_size(worldSizeIni(ws, N, wr)) {
            int color = wr < world_size ? 0 : MPI_UNDEFINED;
            MPI_Comm_split(MPI_COMM_WORLD, color, wr, &world_size_comm);
        }

    void transform(const std::vector<std::complex<real>>& sValues) override;
    void iTransform(const std::vector<std::complex<real>>& fValues) override;

    ~Parallel_MPI_FFT() {
        if (world_rank < world_size)
            MPI_Comm_free(&world_size_comm);
    };

protected:
    void iterativeFFT(std::complex<real> x[], const unsigned int n) override;
    void recursiveFFT(std::complex<real> x[], const unsigned int n) override;

private:
    static int worldSizeIni(int ws_value, const unsigned int N, const int wr){
        // Use a power of 2 ranks:
        const unsigned int pwr = floor(log2(ws_value));
        n_splitting = N >> pwr;

        if (1<<pwr != ws_value && wr == 0)
            std::cout << "Rank number is not a power of two. It'll use only " << (1<<pwr) << " processes." << std::endl;
        return 1 << pwr;
    }



    static bool isRecursive;
    const int world_size; // number of processes
    const int world_rank; // the rank of the process
    static unsigned int n_splitting;
    MPI_Comm world_size_comm;
};

#endif // PARALLEL_MPI_FFT_HPP