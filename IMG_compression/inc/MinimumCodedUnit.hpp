#include "parameters"
#include <iostream>
#include <vector>
#include <complex>
#include <type_traits>

// Eigen library
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra> 

using namespace Eigen;
using Mat = Eigen::MatrixXcd;
using SpMat = Eigen::SparseMatrix<double>;

// 8x8 sub image
class MinimumCodedUnit {
public:
    MinimumCodedUnit(const unsigned int width, const unsigned int height, const unsigned int rowIdx, const unsigned int colIdx)
        : dataWidth( width-colIdx < MCU_SIZE ? width-colIdx : MCU_SIZE )
        , dataHeight( height-rowIdx < MCU_SIZE ? height-rowIdx : MCU_SIZE )
        , imgWidth(width)
        , imgHeight(height)
        , haveFreqValues(false)
        , havePixelsValues(false) {};

    // here FFT, Subtract128 and quantization
    void transform();
    void iTransform();

    void writeCompressedOnFile(std::string &outputFolder, int mcuIdx);
    void readCompressedFromFile(std::string &inputFolder, int mcuIdx);
    
    void readImage(unsigned char* bufferPointer);
    void writeImage(unsigned char* bufferPointer);

protected:

    void FFT2DwithQuantization();

private:

    // Static matrices: 
    int mcuValues[NUM_CHANNELS][MCU_SIZE][MCU_SIZE];
    norm_type normFreqDense[NUM_CHANNELS][8][8];
    phase_type phaseFreqDense[NUM_CHANNELS][8][8];
    int mcuValuesRestored[NUM_CHANNELS][8][8];


    const unsigned int dataWidth;
    const unsigned int dataHeight;
    const unsigned int imgWidth;
    const unsigned int imgHeight;

    // Define compression matrix Q as one static constexpr float mtx 
    static constexpr float Q[MCU_SIZE][MCU_SIZE] = {{16,11,10,16,24,40,51,61},{12,12,14,19,26,58,60,55},{14,13,16,24,40,57,69,56}\
        ,{14,17,22,29,51,87,80,62},{18,22,37,56,68,109,103,77},{24,35,55,64,81,104,113,92},{49,64,78,87,103,121,120,101},{72,92,95,98,112,100,103,99}};
    
    bool haveFreqValues, havePixelsValues;

};