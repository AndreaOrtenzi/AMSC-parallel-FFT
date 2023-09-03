#include "parameters"

#include <iostream>
#include <vector>

#include "../../FFT2D/inc/FFT_2D.hpp"
#include "../../FFT2D_Vec/inc/ParFFT2D.hpp"

// 8x8 sub image
class MinimumCodedUnit {
public:
    MinimumCodedUnit(unsigned char* initialSquare, const unsigned int width, const unsigned int height, const unsigned int rowIdx, const unsigned int colIdx);

    // here FFT, Subtract128 and quantization
    void trasform();
    void iTrasform();

    void writeCompressedOnFile(std::ofstream& writeFilePointer);

protected:

    void FFT2DwithQuantization();

private:

    int mcuValues[NUM_CHANNELS][MCU_SIZE][MCU_SIZE];
    Eigen::SparseMatrix<int, MCU_SIZE, MCU_SIZE> normFreqSparse[NUM_CHANNELS];
    Eigen::Matrix<int> phaseFreqDense[NUM_CHANNELS][MCU_SIZE][MCU_SIZE];


    const unsigned int dataWidth;
    const unsigned int dataHeight;
    const unsigned int imgWidth;
    const unsigned int imgHeight;

    // Define the two version of private member Q, compression matrix: one static constexpr float mtx and Eigen
    static constexpr float Q[MCU_SIZE][MCU_SIZE] = {{16,11,10,16,24,40,51,61},{12,12,14,19,26,58,60,55},{14,13,16,24,40,57,69,56}\
        ,{14,17,22,29,51,87,80,62},{18,22,37,56,68,109,103,77},{24,35,55,64,81,104,113,92},{49,64,78,87,103,121,120,101},{72,92,95,98,112,100,103,99}};
    
    Eigen::Matrix< int,MCU_SIZE, MCU_SIZE> Q_eig;
    Q << 16, 11, 10, 16, 24, 40, 51, 61,
         12, 12, 14, 19, 26, 58, 60, 55,
         14, 13, 16, 24, 40, 57, 69, 56,
         14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77,
         24, 35, 55, 64, 81, 104, 113, 92,
         49, 64, 78, 87, 103, 121, 120, 101,
         72, 92, 95, 98, 112, 100, 103, 99;

};