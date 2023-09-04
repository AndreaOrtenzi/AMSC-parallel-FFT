#include "parameters"

#include <iostream>
#include <vector>

#include "../../FFT2D/inc/FFT_2D.hpp"
#include "../../FFT2D_Vec/inc/ParFFT2D.hpp"

// 8x8 sub image
class MinimumCodedUnit {
public:
    MinimumCodedUnit(unsigned char* initialSquare, const unsigned int width, const unsigned int height, const unsigned int rowIdx, const unsigned int colIdx);
    MinimumCodedUnit()
    : dataWidth( MCU_SIZE )
    , dataHeight( MCU_SIZE )
    , imgWidth(128)
    , imgHeight(128) {
        //std::cout << "Print initial image values: " << std::endl;    
        for(unsigned int channel=0; channel<NUM_CHANNELS; channel++){
            for(unsigned int i = 0; i < MCU_SIZE; i++){
                for(unsigned int j = 0; j < MCU_SIZE; j++){
                    mcuValues[channel][i][j] = i*MCU_SIZE + j;
                    //std::cout << "Initial value in indexes: [" << channel <<"],row[" << i <<"],col[" << j <<"] = " << mcuValues[channel][i][j] << std::endl;
                }
            }
        }
    }
    // here FFT, Subtract128 and quantization
    void transform();
    void iTransform();

    void writeCompressedOnFile(std::ofstream& writeFilePointer);
    void writeImage(unsigned char* bufferPointer);

    const Eigen::Matrix<int, MCU_SIZE, MCU_SIZE>& getnormFreqDenseEigen(){
        return normFreqDenseEigen;
    }

    const Eigen::Matrix<double, MCU_SIZE, MCU_SIZE>& getphaseFreqDenseEigen(){
        return phaseFreqDenseEigen;
    }
    
    const Eigen::Matrix<int, MCU_SIZE, MCU_SIZE>& getmcuValuesRestoredEigen(){
        return mcuValuesRestoredEigen;
    }


protected:

    void FFT2DwithQuantization();

private:

    //Eigen::SparseMatrix<int> normFreqSparse[NUM_CHANNELS];
    //Eigen::Matrix<double, MCU_SIZE, MCU_SIZE> normFreqDense[NUM_CHANNELS];
    //Eigen::Matrix<double, MCU_SIZE, MCU_SIZE> phaseFreqDense[NUM_CHANNELS];

    // Static matrices: 
    int mcuValues[NUM_CHANNELS][MCU_SIZE][MCU_SIZE];
    int normFreqDense[NUM_CHANNELS][8][8];
    double phaseFreqDense[NUM_CHANNELS][8][8];
    int mcuValuesRestored[NUM_CHANNELS][8][8];

    // Eigen Version:
    Eigen::Matrix<double, MCU_SIZE, MCU_SIZE> phaseFreqDenseEigen[NUM_CHANNELS];
    Eigen::Matrix<int, MCU_SIZE, MCU_SIZE> normFreqDenseEigen[NUM_CHANNELS];


    const unsigned int dataWidth;
    const unsigned int dataHeight;
    const unsigned int imgWidth;
    const unsigned int imgHeight;

    // Define the two version of private member Q, compression matrix: one static constexpr float mtx and Eigen version:
    static constexpr float Q[MCU_SIZE][MCU_SIZE] = {{16,11,10,16,24,40,51,61},{12,12,14,19,26,58,60,55},{14,13,16,24,40,57,69,56}\
        ,{14,17,22,29,51,87,80,62},{18,22,37,56,68,109,103,77},{24,35,55,64,81,104,113,92},{49,64,78,87,103,121,120,101},{72,92,95,98,112,100,103,99}};
    
    Eigen::Matrix<double, 8, 8> Q_eig;

};