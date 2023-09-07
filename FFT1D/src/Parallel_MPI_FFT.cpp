#include "../inc/Parallel_MPI_FFT.hpp"
#include <iostream>

bool Parallel_MPI_FFT::isRecursive = false;
unsigned int Parallel_MPI_FFT::n_splitting = 0;

// Perform the Fourier transform using MPI
void Parallel_MPI_FFT::transform(const std::vector<std::complex<real>>& sValues) {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values
    frequencyValues.resize(N);
    
    frequencyValues = sValues;    

    // Check if this rank will participate in the computation:
    if (N <= 1 || world_rank >= world_size){
        std::cout << "Rank " << world_rank << " will not help" << std::endl;
        return;
    }

    std::complex<real> splitted_array[world_size][n_splitting];

    if (world_rank == 0){
        // Split the initial array among ranks:
        for (unsigned int i = 0; i < N; i++){
            splitted_array[i % world_size][i / world_size] = sValues[i];
        }
    }

    // Scatter the data to all ranks:
    if (typeid(real) == typeid(double))
        MPI_Scatter(&splitted_array[0][0], n_splitting, MPI_DOUBLE_COMPLEX, &splitted_array[world_rank][0],
            n_splitting, MPI_DOUBLE_COMPLEX, 0, world_size_comm);
    else MPI_Scatter(&splitted_array[0][0], n_splitting, MPI_COMPLEX, &splitted_array[world_rank][0],
            n_splitting, MPI_COMPLEX, 0, world_size_comm);

    // Compute FFT on this section:
    if (isRecursive){
        if(world_rank == 0)
            std::cout << "--start recursive imp-- world_size:" << world_size << " n_splitting: " << n_splitting << std::endl;

        if (world_size == 1) {
            recursiveFFT(frequencyValues.data(), n_splitting);
        } else {
            recursiveFFT(&splitted_array[world_rank][0], n_splitting);
        }
    } else {
        if(world_rank == 0)
            std::cout << "--start iterative imp--" << std::endl;

        if (world_size == 1) {
            iterativeFFT(frequencyValues.data(), frequencyValues.size());
        } else {
            iterativeFFT(&splitted_array[world_rank][0], n_splitting);
        }
    }

    // Gather the processed data back to rank 0:
    if (typeid(real) == typeid(double))
        MPI_Gather(&splitted_array[world_rank][0], n_splitting, MPI_DOUBLE_COMPLEX, &splitted_array[0][0], n_splitting, MPI_DOUBLE_COMPLEX, 0,
            world_size_comm);
    else MPI_Gather(&splitted_array[world_rank][0], n_splitting, MPI_COMPLEX, &splitted_array[0][0], n_splitting, MPI_COMPLEX, 0,
            world_size_comm);

    if (world_rank == 0) {
        // Finish combining the results of the subproblems:
        unsigned int curr_i = 1;
        unsigned int curr_n = n_splitting;
        unsigned int curr_splitting = world_size >> 1; // /2

        std::complex<real> *ex_x = &splitted_array[0][0]; // array of N elements
        std::complex<real> *curr_x = static_cast<std::complex<double>*>(malloc(N*sizeof(std::complex<real>)));

        while (curr_n < N) {
            
            for (unsigned int j = 0; j < curr_splitting; j++){
                for (unsigned int i = 0; i < curr_n; i++) {

                    std::complex<real> t((std::complex<real>)std::polar(1.0, -M_PI*i/curr_n) * ex_x[j*curr_n + i + curr_splitting*curr_n]);
                    if (curr_n != N >> 1) {
                        curr_x[ j*curr_n*curr_splitting + i] = ex_x[j*curr_n + i] + t;
                        curr_x[ j*curr_n*curr_splitting + i + curr_n] = ex_x[j*curr_n + i] - t;
                    } else {
                        std::complex<real> *fqV = frequencyValues.data();
                        fqV[ curr_splitting*j*curr_n + i] = ex_x[ j*curr_n + i] + t;
                        fqV[ curr_splitting*j*curr_n + i + curr_n] = ex_x[ j*curr_n + i] - t;
                    }                        
                }
            }
            
            std::complex<real> *temp = ex_x;
            ex_x = curr_x;
            curr_x = temp;

            curr_splitting = curr_splitting >> 1;
            curr_n = curr_n << 1;
            curr_i++;
        }
    }

    isRecursive = !isRecursive;
}

// A parallel implementation of the FFT recursive method using MPI.
void Parallel_MPI_FFT::recursiveFFT(std::complex<real> x[], const unsigned int n) {
    if (n <= 1) {
        return;
    }
    const unsigned int halfn = n >> 1;
    std::complex<real> even[halfn], odd[halfn];
    for (unsigned int i = 0; i < halfn; i++) {
        even[i] = x[2*i];
        odd[i] = x[2*i+1];
    }
    recursiveFFT(even, halfn);
    recursiveFFT(odd, halfn);
    // Combine the results of the two subproblems:
    for (unsigned int i = 0; i < halfn; i++) {
        std::complex<real> t((std::complex<real>)std::polar(1.0, -2*M_PI*i/n) * odd[i]);
        x[i] = even[i] + t;
        x[i+halfn] = even[i] - t;
    }
}

// A parallel implementation of the FFT iterative method using MPI.
void Parallel_MPI_FFT::iterativeFFT(std::complex<real> x[], const unsigned int n) {
    unsigned int numBits = static_cast<unsigned int>(log2(n));
    for (unsigned int i = 0; i < n; i++) 
    {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((i >> k) & 1U);
        }
        if (j > i) {
            std::swap(x[i], x[j]);
        }
    }
    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<real> wm = std::exp(-2.0 * M_PI * std::complex<real>(0, 1) / static_cast<real>(m));
        for (unsigned int k = 0; k < n; k += m) {
            std::complex<real> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<real> t = w * x[k + j + m / 2];
                std::complex<real> u = x[k + j];
                x[k + j] = u + t;
                x[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
}

// Perform the inverse Fourier transform using MPI
void Parallel_MPI_FFT::iTransform(const std::vector<std::complex<real>>& fValues) {
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    spatialValues.resize(N);

    std::vector<std::complex<real>> freqVec = fValues;
    unsigned int numBits = static_cast<unsigned int>(log2(N));

    // Bit reversal:
    for (unsigned int l = 0; l < N; l++) {
        unsigned int j = 0;
        for (unsigned int k = 0; k < numBits; k++) {
            j = (j << 1) | ((l >> k) & 1U);
        }
        if (j > l) {
            std::swap(freqVec[l], freqVec[j]);
        }
    }
    
    for (unsigned int s = 1; s <= numBits; s++) {
        unsigned int m = 1U << s; 
        std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
        for (unsigned int k = 0; k < N; k += m) {
            std::complex<double> w = 1.0;
            for (unsigned int j = 0; j < m / 2; j++) {
                std::complex<double> t = w * freqVec[k + j + m / 2];
                std::complex<double> u = freqVec[k + j];
                freqVec[k + j] = u + t;
                freqVec[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }

    // Normalize by dividing by N
    for (unsigned int i = 0; i < N; ++i) {
        spatialValues[i] = freqVec[i] / static_cast<real>(N);
    }   
}
