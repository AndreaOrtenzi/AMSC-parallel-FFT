#include <cmath>
#include <algorithm> // for usage of std::for_each
#include <filesystem> // for create directory compressed_images
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
    constexpr int numChannels = NUM_CHANNELS;
    unsigned char *imageData = stbi_load(inputFilePath.c_str(), &imgWidth, &imgHeight, &numChannels, 0);

    if (!imageData) {
        std::cerr << "Error occurred during the image reading." << std::endl;
        return;
    }

    // Calculate number of MCUs in width and in height:
    unsigned int numMCUsWidth = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;

    // Create MCUs from image's pixel:
    for (unsigned int row = 0; row < numMCUsHeight; row++) {
        for (unsigned int col = 0; col < numMCUsWidth; col++) {
            unsigned int startX = col * MCU_SIZE;
            unsigned int startY = row * MCU_SIZE;

            // Let's create a MCU from pixel in the specified area:
            MinimumCodedUnit mcu(&imageData[(startY * imgWidth + startX) * numChannels], imgWidth, imgHeight, startY, startX);

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

    // int width, height, numChannels = NUM_CHANNELS;
    // unsigned char *imageData = stbi_load(inputFilePath.c_str(), &width, &height, &numChannels, 0);

    // if (!imageData) {
    //     std::cerr << "Error occurred during the reading of compressed image." << std::endl;
    //     return;
    // }

    /* ------------------------------------------------------------------
    ---------------------implementa la lettura qua ---------------------------
    ---------------------------------------------------------------------*/
    // Free the memory:
    stbi_image_free(imageData);
}

void Image::writeCompressed() {

    std::filesystem::path outputPath = "../compressed_images";

    // If directory doesn't exist, create it: 
    if (!std::filesystem::exists(outputPath)) {
        std::filesystem::create_directory(outputPath);
    }
    
    for ( unsigned int i= 0; i< imageMCUs.size(); ++i)
        imageMCUs[i].writeCompressedOnFile(outputPath, i);

}

void Image::writeImage(){

    // Create a byte array for the final image:
    std::vector<unsigned char> imageBuffer(imgWidth * imgHeight * NUM_CHANNELS);

    // Converts the restored value arrays to a byte array for the image:
    // for (unsigned int mcuRow = 0; mcuRow < imgHeight / MCU_SIZE; mcuRow++) {
    //     for (unsigned int mcuCol = 0; mcuCol < imgWidth / MCU_SIZE; mcuCol++) {
            
    unsigned int numMCUsWidth = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;

    for (unsigned int row = 0; row < numMCUsHeight; row++) {
        for (unsigned int col = 0; col < numMCUsWidth; col++) {
            unsigned int startX = col * MCU_SIZE;
            unsigned int startY = row * MCU_SIZE;
            unsigned int mcuIdx = row * (imgWidth / MCU_SIZE) + col;
            MinimumCodedUnit& mcu = imageMCUs[mcuIdx];

            mcu.writeImage(&imageData[(startY * imgWidth + startX) * NUM_CHANNELS]);

        }
    }

    // stb_image_write to write finale JPEG image:
    stbi_write_jpg(outputFileName.c_str(), imgWidth, imgHeight, NUM_CHANNELS, imageBuffer.data(), QUALITY);
}





