#include <cmath>
#include <algorithm> // per std::for_each
#include "../inc/parameters"
#include "../inc/Image.hpp"

// Eigen library
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/SparseExtra> 

using namespace std;
using namespace Eigen;
using Mat = Eigen::MatrixXcd;
using SpMat = Eigen::SparseMatrix<double>;

#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"


Image::Image(std::string inputFilePath, std::string outputFilePath, bool isInputCompressed)
    : inputFilePath(inputFilePath)
{
    /*------------ inizializzazioni di classe qua---------------------------
    ------------------------------------------------------------------------ */ 

    // Call to the right function:
    if (isInputCompressed) {
        readCompressed();
    } else {
        readImage();
    }

    /*------------fare altre inizializzazioni di classe qua-----------------
    ------------------------------------------------------------------------ */ 
}

void Image::readImage(){
    int width, height, numChannels = NUM_CHANNELS;
    unsigned char *imageData = stbi_load(inputFilePath.c_str(), &width, &height, &numChannels, 0);

    if (!imageData) {
        std::cerr << "Error occurred during the image reading." << std::endl;
        return;
    }

    // Calculate number of MCUs in width and in height:
    unsigned int numMCUsWidth = (width + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (height + MCU_SIZE - 1) / MCU_SIZE;

    // Create MCUs from image's pixel:
    for (unsigned int row = 0; row < numMCUsHeight; row++) {
        for (unsigned int col = 0; col < numMCUsWidth; col++) {
            unsigned int startX = col * MCU_SIZE;
            unsigned int startY = row * MCU_SIZE;

            // Let's create a MCU from pixel in the specified area:
            MinimumCodedUnit mcu(&imageData[(startY * width + startX) * numChannels], width, height, startY, startX);

            // Add the new unit to MCUs' vector of image:
            imageMCUs.push_back(mcu);
        }
    }

    // Free the memory used for image: 
    stbi_image_free(imageData);
}

// Trasform every MCU in the vector imageMCUs: 
void Image::trasform(){

    // Define a lambda function to apply FFT2D for each MCU: [] --> captures nothing
    auto apply_FFT2D = [](MinimumCodedUnit &mcu) {
        mcu.trasform(); // call to trasform method of MinimumCodeUnit
    };

    // Use of std::for_each to apply lambda function to every MCU in the vector:
    std::for_each(imageMCUs.begin(), imageMCUs.end(), apply_FFT2D);
}

//Inverse Trasform every MCU in the vector imageMCUs:
void Image::iTrasform(){
    
    // At the same way for trasform method, define a lambda function:
    auto apply_iFFT2D  = [](MinimumCodedUnit &mcu) {
        mcu.iTrasform(); // call to iTrasform method of MinimumCodeUnit
    };

    // For each MCU apply the lambda function just definied: 
    std::for_each(imageMCUs.begin(), imageMCUs.end(), apply_iFFT2D);

}
// 
void Image::readCompressed(){

    int width, height, numChannels = NUM_CHANNELS;
    unsigned char *imageData = stbi_load(inputFilePath.c_str(), &width, &height, &numChannels, 0);

    if (!imageData) {
        std::cerr << "Error occurred during the reading of compressed image." << std::endl;
        return;
    }

    /* ------------------------------------------------------------------
    ---------------------implementa la lettura qua ---------------------------
    ---------------------------------------------------------------------*/
    // Free the memory:
    stbi_image_free(imageData);
}

// scrivere il file per l'immagine compressa (saveMarket di Eigen per tutte le matrici e poi ricomponi l'immagine)--> da fare in writeCompressedOnFile
void Image::writeCompressed() {
    // Open file binary to write:
    std::ofstream outputFile(outputFilePath.c_str(), std::ios::binary);

    if (!outputFile.is_open()) {
        std::cerr << "Error opening output file." << std::endl;
        return;
    }

    // For each MCU in the vector: 
    for (const MinimumCodedUnit &mcu : imageMCUs) {
        // Call to writeCompressedOnFile method passing the file pointer:
        mcu.writeCompressedOnFile(outputFile);
    }

    // Close the file:
    outputFile.close();
}




