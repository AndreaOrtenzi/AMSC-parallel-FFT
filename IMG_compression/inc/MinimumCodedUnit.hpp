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

// Define the class for an 8x8 Minimum Coded Unit (MCU) sub-image
class MinimumCodedUnit {
public:
    MinimumCodedUnit(const unsigned int width, const unsigned int height, const unsigned int rowIdx, const unsigned int colIdx)
        : dataWidth( width-colIdx < MCU_SIZE ? width-colIdx : MCU_SIZE )    // Calculate the width of MCU data
        , dataHeight( height-rowIdx < MCU_SIZE ? height-rowIdx : MCU_SIZE ) // Calculate the height of MCU data
        , imgWidth(width)           // Store the width of the image
        , imgHeight(height)         // Store the height of the image
        , haveFreqValues(false)     // Initialize flags for frequency and pixel values
        , havePixelsValues(false) {};

    // Methods for performing transformations and inverse transformations
    void transform(); // Perform transformations (e.g., FFT, Subtract128, quantization)
    void iTransform(); // Perform inverse transformations

    // Methods for writing and reading compressed data to/from files
    void writeCompressedOnFile(std::string &outputFolder, int mcuIdx); // Write compressed data to a file
    void readCompressedFromFile(std::string &inputFolder, int mcuIdx); // Read compressed data from a file
    
    void readImage(unsigned char* bufferPointer);
    void writeImage(unsigned char* bufferPointer);

protected:

    void FFT2DwithQuantization();

private:
    // Static matrices and constants
    int mcuValues[NUM_CHANNELS][MCU_SIZE][MCU_SIZE];    // Array to store MCU values
    norm_type normFreqDense[NUM_CHANNELS][8][8];        // Array to store normalized frequency domain values
    phase_type phaseFreqDense[NUM_CHANNELS][8][8];      // Array to store phase frequency domain values
    int mcuValuesRestored[NUM_CHANNELS][8][8];          // Array to store restored MCU values


    const unsigned int dataWidth; // Width of data without padding
    const unsigned int dataHeight; // Height of data without padding
    const unsigned int imgWidth; // Width of the image in pixel 
    const unsigned int imgHeight; // Height of the image in pixel

    // Define compression matrix Q as a static constexpr float mtx 
    static constexpr float Q[MCU_SIZE][MCU_SIZE] = {{16,11,10,16,24,40,51,61},{12,12,14,19,26,58,60,55},{14,13,16,24,40,57,69,56}\
        ,{14,17,22,29,51,87,80,62},{18,22,37,56,68,109,103,77},{24,35,55,64,81,104,113,92},{49,64,78,87,103,121,120,101},{72,92,95,98,112,100,103,99}};
    
    bool haveFreqValues, havePixelsValues; // Flags for frequency and pixel values

};