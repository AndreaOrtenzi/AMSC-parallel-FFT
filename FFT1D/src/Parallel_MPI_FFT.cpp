#include "../inc/Parallel_MPI_FFT.hpp"
#include <iostream>

bool Parallel_MPI_FFT::isRecursive = false;
unsigned int Parallel_MPI_FFT::n_splitting = 0;

// Perform the Fourier transform using MPI
void Parallel_MPI_FFT::transform(const std::vector<std::complex<real>>& sValues) {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values

    // Check if this rank will participate in the computation:
    if (N <= 1 || world_rank >= world_size){
        std::cout << "Rank " << world_rank << " will not help" << std::endl;
        return;
    }

    std::complex<real> splitted_array[n_splitting];

    if (world_rank == 0){
        frequencyValues.resize(N);

        // Reorder the initial vector to allow each ranks to do normal FFT on a  piece of it:
        unsigned int w = 0;
        for (unsigned int i = 0; i < world_size; ++i) {
            for (unsigned int j = 0; j < n_splitting; ++j) {
                frequencyValues[w] = sValues[ j * world_size + i ];
                w++;
            }
        }
    }
    
    // Scatter the data to all ranks:
    if (typeid(real) == typeid(double))
        MPI_Scatter(frequencyValues.data(), n_splitting, MPI_DOUBLE_COMPLEX, splitted_array,
            n_splitting, MPI_DOUBLE_COMPLEX, 0, world_size_comm);
    else MPI_Scatter(frequencyValues.data(), n_splitting, MPI_COMPLEX, splitted_array,
            n_splitting, MPI_COMPLEX, 0, world_size_comm);

    // Compute FFT on this section:
    if (isRecursive){
        if(world_rank == 0)
            std::cout << "--start recursive imp-- world_size:" << world_size << " n_splitting: " << n_splitting << std::endl;

        recursiveFFT(splitted_array, n_splitting);

    } else {
        if(world_rank == 0)
            std::cout << "--start iterative imp--" << std::endl;

        
        iterativeFFT(splitted_array, n_splitting);
    }

    // Gather the processed data back to rank 0:
    if (typeid(real) == typeid(double))
        MPI_Gather(splitted_array, n_splitting, MPI_DOUBLE_COMPLEX, frequencyValues.data(), n_splitting, MPI_DOUBLE_COMPLEX, 0,
            world_size_comm);
    else MPI_Gather(splitted_array, n_splitting, MPI_COMPLEX, frequencyValues.data(), n_splitting, MPI_COMPLEX, 0,
            world_size_comm);

    // Finish combining the results of the subproblems:
    if (world_rank == 0) {
        std::complex<real> *to_do_free;
        
        unsigned int curr_i = 1;
        unsigned int curr_n = n_splitting;
        unsigned int curr_splitting = world_size >> 1; // /2

        std::complex<real> *ex_x = frequencyValues.data(); // array of N elements
        std::complex<real> *curr_x = static_cast<std::complex<double>*>(malloc(N*sizeof(std::complex<real>)));
        to_do_free = curr_x;

        while (curr_n < N) {
            
            for (unsigned int j = 0; j < curr_splitting; j++){
                for (unsigned int i = 0; i < curr_n; i++) {

                    std::complex<real> t((std::complex<real>)std::polar(1.0, -M_PI*i/curr_n) * ex_x[j*curr_n + i + curr_splitting*curr_n]);

                    curr_x[ j*curr_n*curr_splitting + i] = ex_x[j*curr_n + i] + t;
                    curr_x[ j*curr_n*curr_splitting + i + curr_n] = ex_x[j*curr_n + i] - t;                    
                }
            }
            
            std::complex<real> *temp = ex_x;
            ex_x = curr_x;
            curr_x = temp;

            curr_splitting = curr_splitting >> 1;
            curr_n = curr_n << 1;
            curr_i++;
        }
        // Copy the results in frequency vector
        if (ex_x != frequencyValues.data()) {
            for (unsigned int i = 0; i < N; ++i) {
                frequencyValues[i] = ex_x[i];
            }
        }

        free(to_do_free);
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

void iFFT(std::complex<real> *freqVec, const unsigned int n_splitting) {
    // Perform the inverse Fourier transform on the frequency values and store the result in the spatial values
    const unsigned int numBits = static_cast<unsigned int>(log2(n_splitting));

    // Bit reversal:
    for (unsigned int l = 0; l < n_splitting; l++) {
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
        for (unsigned int k = 0; k < n_splitting; k += m) {
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
}

// Perform the inverse Fourier transform using MPI
void Parallel_MPI_FFT::iTransform(const std::vector<std::complex<real>>& fValues) {
    // Perform the Fourier transform on the spatial values and store the result in the frequency values

    // Check if this rank will participate in the computation:
    if (N <= 1 || world_rank >= world_size){
        std::cout << "Rank " << world_rank << " will not help" << std::endl;
        return;
    }

    std::complex<real> splitted_array[n_splitting];

    if (world_rank == 0){
        spatialValues.resize(N);

        // Reorder the initial vector to allow each ranks to do normal FFT on a  piece of it:
        unsigned int w = 0;
        for (unsigned int i = 0; i < world_size; ++i) {
            for (unsigned int j = 0; j < n_splitting; ++j) {
                spatialValues[w] = fValues[ j * world_size + i ];
                w++;
            }
        }
    }
    
    // Scatter the data to all ranks:
    if (typeid(real) == typeid(double))
        MPI_Scatter(spatialValues.data(), n_splitting, MPI_DOUBLE_COMPLEX, splitted_array,
            n_splitting, MPI_DOUBLE_COMPLEX, 0, world_size_comm);
    else MPI_Scatter(spatialValues.data(), n_splitting, MPI_COMPLEX, splitted_array,
            n_splitting, MPI_COMPLEX, 0, world_size_comm);

    iFFT(splitted_array, n_splitting);
    

    // Gather the processed data back to rank 0:
    if (typeid(real) == typeid(double))
        MPI_Gather(splitted_array, n_splitting, MPI_DOUBLE_COMPLEX, spatialValues.data(), n_splitting, MPI_DOUBLE_COMPLEX, 0,
            world_size_comm);
    else MPI_Gather(splitted_array, n_splitting, MPI_COMPLEX, spatialValues.data(), n_splitting, MPI_COMPLEX, 0,
            world_size_comm);
    
    // Finish combining the results of the subproblems:
    if (world_rank == 0) {
        std::complex<real> *to_do_free;
        
        unsigned int curr_i = 1;
        unsigned int curr_n = n_splitting;
        unsigned int curr_splitting = world_size >> 1; // /2

        std::complex<real> *ex_x = spatialValues.data(); // array of N elements
        std::complex<real> *curr_x = static_cast<std::complex<double>*>(malloc(N*sizeof(std::complex<real>)));
        to_do_free = curr_x;

        while (curr_n < N) {
            
            for (unsigned int j = 0; j < curr_splitting; j++){
                for (unsigned int i = 0; i < curr_n; i++) {

                    std::complex<real> t((std::complex<real>)std::polar(1.0, M_PI*i/curr_n) * ex_x[j*curr_n + i + curr_splitting*curr_n]);

                    curr_x[ j*curr_n*curr_splitting + i] = ex_x[j*curr_n + i] + t;
                    curr_x[ j*curr_n*curr_splitting + i + curr_n] = ex_x[j*curr_n + i] - t;                    
                }
            }
            
            std::complex<real> *temp = ex_x;
            ex_x = curr_x;
            curr_x = temp;

            curr_splitting = curr_splitting >> 1;
            curr_n = curr_n << 1;
            curr_i++;
        }
        // Copy the results in frequency vector
        for (unsigned int i = 0; i < N; ++i) {
            spatialValues[i] = ex_x[i] / static_cast<real>(N);
        }
        

        free(to_do_free);
    }
}
