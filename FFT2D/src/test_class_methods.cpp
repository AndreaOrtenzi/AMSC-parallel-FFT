#include "../inc/FFT_2D.hpp"
#include <iostream>
#include <iomanip>
#include <random>
#include <chrono>
#ifndef ROW_LENGTH
#define POW 10
#endif
#ifndef MAX_MAT_VALUES
#define MAX_MAT_VALUES 255
#endif
#ifndef CHECK_CORRECTNESS
#define CHECK_CORRECTNESS true
#endif

bool checkCorrectness(const Mat &correct, const Mat &toCheck) {
    constexpr double eps(1e-10 * MAX_MAT_VALUES);

    for (int i = 0; i < correct.rows(); ++i) {
        for (int j = 0; j < correct.cols(); ++j) {
            const std::complex<double> &correctValue = correct.coeff(i, j);
            const std::complex<double> &toCheckValue = toCheck.coeff(i, j);

            if (std::abs(correctValue.imag() - toCheckValue.imag()) > eps ||
                std::abs(correctValue.real() - toCheckValue.real()) > eps) {
                std::cout << "Problem with element at (" << i << ", " << j << "): " << toCheckValue
                          << ", It should be: " << correctValue << endl;
                std::cout << "WRONG TRANSFORMATION!" << std::endl;          
                return false;
            }
        }
    }

    std::cout << "Correct transformation!" << std::endl;
    return true;
}

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

int main() {
    // Impostiamo il seed per il generatore di numeri casuali
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 255.0);

    // Impostiamo le dimensioni della matrice (2^pow x 2^pow)
    unsigned int size = std::pow(2, POW);

    // Creiamo la matrice di input con numeri interi casuali
    Mat inputMatrix(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            inputMatrix(i, j) = rand() % MAX_MAT_VALUES;
        }
    }
    // Crea copie della inputMatrix per confronti:
    Mat copiaInput(size, size), copiaInput2(size,size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            copiaInput(i, j) = inputMatrix(i, j);
            copiaInput2(i, j)     = inputMatrix(i, j);
        }
    }
        // std::cout << "Space values:" << std::endl;
        // for (int j = 0; j < copiaInput.rows(); ++j) 
        // {
        //     for (int k = 0; k < copiaInput.cols(); ++k) {
        //         std::complex<double> value = copiaInput(j, k);
        //         std::cout << "\t(" << std::fixed << std::setprecision(3) << value.real() << ", " << value.imag() << ")";
        //     }       
        // std::cout << std::endl;
        // }
    std::cout << "Input matrix filled with "<< size << " x " << size << " elements." << std::endl;
    std::cout << "*****************************************************************************" <<std::endl;

    Mat freqiterseq(size,size);
    // Creiamo un'istanza di FFT_2D
    FFT_2D fft1(inputMatrix, freqiterseq);

    // Testiamo il metodo di trasformazione sequenziale
    auto startSeq = std::chrono::high_resolution_clock::now();
    fft1.transform_seq();
    auto endSeq = std::chrono::high_resolution_clock::now();
    auto durationSeq = std::chrono::duration_cast<std::chrono::microseconds>(endSeq - startSeq).count();
    Mat freqSeq = fft1.getFrequencyValues();
    // std::cout << "Frequency values after sequential transform:" << std::endl;
    // for (int j = 0; j < freqSeq.rows(); ++j) {
    //     for (int k = 0; k < freqSeq.cols(); ++k) {
    //         std::complex<double> value = freqSeq(j, k);
    //         std::cout << "\t(" << std::fixed << std::setprecision(3) << value.real() << ", " << value.imag() << ")";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "Sequential Transform Time: " << durationSeq << " microseconds" << std::endl;
    std::cout << "**********************************************************************************************************" <<std::endl;

    // Testiamo il metodo di trasformazione parallela
    Mat freqiterpar(size, size);
    FFT_2D fft2(copiaInput, freqiterpar);
    auto startPar = std::chrono::high_resolution_clock::now();
    fft2.transform_par();
    auto endPar = std::chrono::high_resolution_clock::now();
    auto durationPar = std::chrono::duration_cast<std::chrono::microseconds>(endPar - startPar).count();
    Mat freqPar = fft2.getFrequencyValues();
    // std::cout << "Frequency values after parallel transform:" << std::endl;
    // for (int j = 0; j < freqPar.rows(); ++j) {
    //     for (int k = 0; k < freqPar.cols(); ++k) {
    //         std::complex<double> value = freqPar(j, k);
    //         std::cout << "\t(" << std::fixed << std::setprecision(3) << value.real() << ", " << value.imag() << ")";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "Parallel Transform Time: " << durationPar << " microseconds" << std::endl;
    std::cout << "**********************************************************************************************************" <<std::endl;

    // Testiamo il metodo di trasformazione inversa
    auto startinv = std::chrono::high_resolution_clock::now();
    fft2.iTransform();
    auto endinv = std::chrono::high_resolution_clock::now();
    auto durationInv = std::chrono::duration_cast<std::chrono::microseconds>(endinv - startinv).count();

    Mat spatialValues = fft2.getSpatialValues();
    // std::cout << "Spatial values after inverse transform:" << std::endl;
    // for (int j = 0; j < spatialValues.rows(); ++j) {
    //     for (int k = 0; k < spatialValues.cols(); ++k) {
    //         std::complex<double> value = spatialValues(j, k);
    //         std::cout << "\t(" << std::fixed << std::setprecision(3) << value.real() << ", " << value.imag() << ")";
    //     }
    //     std::cout << std::endl;
    // }
    std::cout << "iFFT Time: " << durationInv << " microseconds" << std::endl;
    //Check correctness of inverse method:
    #if CHECK_CORRECTNESS
        checkCorrectness(copiaInput2, spatialValues);
    #endif
    std::cout << "**********************************************************************************************************" <<std::endl;

    return 0;
}
