#include <iostream>
#include <vector>
#include <complex>
#include <string.h>
// #include "../inc/FFT_2D.hpp"

#define MAX_MAT_VALUES 250
#define ROW_LENGTH 4

template <class T>
void DFT_2D(const std::vector<std::vector<T>> &spatialValues, std::vector<std::vector<std::complex<double>>> &frequencyValues) {
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

template <class T>
int checkCorrectness(const std::string implemName, const std::vector<std::vector<T>> &correct, const std::vector<std::vector<T>> &toCheck) {
    bool isCorrect = true;
    constexpr double eps(1e-10 * MAX_MAT_VALUES);

    for (int i = 0; i < correct.size(); ++i) {
        for (int j = 0; j < correct[0].size(); ++j) {
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
            matToFill[i][j] = static_cast<T>(rand() % MAX_MAT_VALUES);
        }
    }
}


template <class C> 
void iterative_sequential(std::vector<std::vector<C>>& input_matrix, // togliere const a input_matrix, ma prima controllare no const solo x swap
     std::vector<std::vector<std::complex<double>>>& freq_matrix){

    if (input_matrix.empty())
        return;
        
    const unsigned int n_cols = input_matrix[0].size(), n_rows = input_matrix.size();

    unsigned int numBits = static_cast<unsigned int>(log2(n_cols));

    std::vector<std::complex<double>> col(n_rows,0.0);
    std::vector<std::vector<std::complex<double>>> input_cols(n_cols,col);
    freq_matrix.resize(n_rows);
    
    //First pass: Apply FFT to each row
    for (unsigned int i = 0; i < n_rows; ++i) {
        std::vector<std::complex<double>> &row_vector = freq_matrix[i];
        row_vector.resize(n_cols);

        
        for (unsigned int l = 0; l < n_cols; l++) { // **************
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(input_matrix[i][l], input_matrix[i][ji]);
            }
        }
        // use last iteration to write column vectors and the first to not override input_matrix
        // s = 1
        {
            unsigned int m = 1U << 1; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {

                // unsigned int ji = 0;
                // for (unsigned int l = 0; l < numBits; l++) {
                //     ji = (ji << 1) | ((k >> l) & 1U);
                // }
                // if (ji > k) {
                //     std::swap(input_matrix[i][k], input_matrix[i][ji]);
                // }

                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * static_cast<std::complex<double>>(input_matrix[i][k + j + m / 2]);
                    std::complex<double> u = static_cast<std::complex<double>>(input_matrix[i][k + j]);
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // swap again to restore original input_matrix
        
        for (unsigned int l = 0; l < n_cols; l++) { // **************
            unsigned int ji = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                ji = (ji << 1) | ((l >> k) & 1U);
            }
            if (ji > l) {
                std::swap(input_matrix[i][l], input_matrix[i][ji]);
            }
        }

        for (unsigned int s = 2; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    row_vector[k + j] = u + t;
                    row_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_cols; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * row_vector[k + j + m / 2];
                    std::complex<double> u = row_vector[k + j];
                    input_cols[k + j][i] = u + t;
                    input_cols[k + j + m / 2][i] = u - t;
                    w *= wm;
                }
            }
        }

        
        // input_matrix.row(i) = row_vector;
    }

    //Second pass: Apply FFT to each column
    numBits = static_cast<unsigned int>(log2(n_rows));
    for (unsigned int i = 0; i < n_cols; ++i) {
        std::vector<std::complex<double>> &col_vector = input_cols[i];
        
        for (unsigned int l = 0; l < n_rows; l++){
            unsigned int j = 0;
            for (unsigned int k = 0; k < numBits; k++) {
                j = (j << 1) | ((l >> k) & 1U);
            }
            if (j > l) {
                std::swap(col_vector[l], col_vector[j]);
            }
        }

        for (unsigned int s = 1; s < numBits; s++) {
            unsigned int m = 1U << s; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    col_vector[k + j] = u + t;
                    col_vector[k + j + m / 2] = u - t;
                    w *= wm;
                }
            }
        }
        // s == numBits
        {
            unsigned int m = 1U << numBits; 
            std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
            for (unsigned int k = 0; k < n_rows; k += m) {
                std::complex<double> w = 1.0;
                for (unsigned int j = 0; j < m / 2; j++) {
                    std::complex<double> t = w * col_vector[k + j + m / 2];
                    std::complex<double> u = col_vector[k + j];
                    freq_matrix[k + j][i] = u + t;
                    freq_matrix[k + j + m / 2][i] = u - t;
                    w *= wm;
                }
            }
        }
    }
}

unsigned int print_i = 0;
template <class T>
void printInt(std::vector<std::vector<T>> vec){
    std::cout << "Print " << print_i << std::endl;
    for (auto i : vec){
        for (auto j : i) {
            std::cout << static_cast<int>(j) << " ";
        }
        std::cout << std::endl;
    }
}


int main(int argc, char *argv[]) {
    const unsigned int pow = std::log2(ROW_LENGTH);
    std::vector<std::vector<unsigned char>> vecXSpace;
    std::vector<std::vector<std::complex<double>>> vecXFreq;
    fill_input_matrix(vecXSpace, pow);

    printInt(vecXSpace);

    // copy to another vector to check is not overrided:
    std::vector<std::vector<unsigned char>> vecXSpaceCPY;
    vecXSpaceCPY.resize(vecXSpace.size());
    for(unsigned int i = 0; i< vecXSpace.size(); i++){
        vecXSpaceCPY[i] = vecXSpace[i];
    }

    iterative_sequential(vecXSpace,vecXFreq);

    // check is not modified:
    checkCorrectness("iterative_sequential don't modify vecXSpace", vecXSpaceCPY, vecXSpace);

    std::vector<std::vector<std::complex<double>>> vecXFreqCorr;
    DFT_2D(vecXSpace, vecXFreqCorr);// ,xSpace.rows())

    checkCorrectness("DFT_2D don't modify vecXSpace", vecXSpaceCPY, vecXSpace);

    checkCorrectness("Name", vecXFreqCorr, vecXFreq);
}