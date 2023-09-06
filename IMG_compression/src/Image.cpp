#include "../inc/Image.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"

/*
* paths to folder end with the / character
* jpegImgsFolderPath: is the path to the folder that contains the jpeg images (initial and restored)
* encodedFolderPath: is the path to the folder where will be created a folder with all the encoded files
* imgName: image name corresponds to the file name without extension and the folder name in encodedFolderPath (will be created)
* isInputCompressed: true if it has to get the pixel informations from encoded files, false from jpeg file
*/
Image::Image(std::string jpegImgsFolderPath_, std::string encodedFolderPath_, std::string imgName_, bool isInputCompressed = false)
    : jpegImgsFolderPath(jpegImgsFolderPath_)
    , encodedFolderPath(encodedFolderPath_)
    , imgName(imgName_)
    , hasFreqValues(isInputCompressed)
    , hasPixelsValues(!isInputCompressed)
{
    // Check if folders and files exist:
    if (!std::filesystem::is_directory(jpegImgsFolderPath) || !std::filesystem::is_directory(encodedFolderPath)) {
        std::cerr << "One of the specified folders doesn't exist." << std::endl;
        throw 1;
    }

    // Call to the right function:
    if (isInputCompressed) {
        readCompressed();
    } else {
        std::string imagePath = jpegImgsFolderPath + imgName + ".jpg";
        if (!std::filesystem::exists(imagePath)) {
            std::cerr << "Image file doesn't exist." << std::endl;
            throw 1;
        }
        readImage();
    }
}

void Image::readImage(){
    int numChannels = NUM_CHANNELS;
    std::string imagePath = jpegImgsFolderPath + imgName + ".jpg";
    unsigned char *imageData = stbi_load(imagePath.c_str(), &imgWidth, &imgHeight, &numChannels, 0);

    if (!imageData) {
        std::cerr << "Error occurred during the image reading." << std::endl;
        throw 1;
    }
    std::cout << "**** Starting Reading Image  ****" << std::endl;
    // Calculate number of MCUs in width and in height:
    unsigned int numMCUsWidth = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;

    // Create MCUs from image's pixel:
    for (unsigned int row = 0; row < numMCUsHeight; row++) {
        for (unsigned int col = 0; col < numMCUsWidth; col++) {
            unsigned int startX = col * MCU_SIZE;
            unsigned int startY = row * MCU_SIZE;

            // Let's create a MCU from pixel in the specified area:
            MinimumCodedUnit mcu(imgWidth, imgHeight, startY, startX);
            mcu.readImage(&imageData[(startY * imgWidth + startX) * numChannels]);

            // Add the new unit to MCUs' vector of image:
            imageMCUs.push_back(mcu);
        }
    }

    // Free the memory used for image: 
    stbi_image_free(imageData);

    std::cout << "**** Read Image Finished ****" << std::endl;

    hasPixelsValues = true;
}

// Trasform every MCU in the vector imageMCUs: 
void Image::trasform(){

    std::cout << "**** Starting FFT ****" << std::endl;

    if (!hasPixelsValues){
        std::cerr << "There are not pixels values here!" << std::endl;
        throw 1;
    }

    // Define a lambda function to apply FFT2D for each MCU: [] --> captures nothing
    auto apply_FFT2D = [](MinimumCodedUnit &mcu) {
        mcu.transform(); // call to trasform method of MinimumCodeUnit
    };

    // Use of std::for_each to apply lambda function to every MCU in the vector:
    std::for_each(imageMCUs.begin(), imageMCUs.end(), apply_FFT2D);

    hasFreqValues = true;

    std::cout << "**** Finished FFT ****" << std::endl;
}

//Inverse Trasform every MCU in the vector imageMCUs:
void Image::iTrasform(){

    if (!hasFreqValues){
        std::cerr << "There are not frequency values here!" << std::endl;
        throw 2;
    }

    std::cout << "**** Starting iFFT ****" << std::endl;
    
    // At the same way for trasform method, define a lambda function:
    auto apply_iFFT2D  = [](MinimumCodedUnit &mcu) {
        mcu.iTransform(); // call to iTrasform method of MinimumCodeUnit
    };

    // For each MCU apply the lambda function just definied: 
    std::for_each(imageMCUs.begin(), imageMCUs.end(), apply_iFFT2D);

    hasPixelsValues = true;

    std::cout << "**** iFFT finished ****" << std::endl;
}

void Image::readCompressed(){

    std::string outputFolderPath = encodedFolderPath + imgName;
    std::cout << "**** Starting Reading Compressed ****" << std::endl;

    {
        std::string inFile = outputFolderPath + "/metadata.txt";
        std::ifstream rfile(inFile);

        if (!rfile.is_open()) {
            std::cerr << "Failed to open the file: " << inFile << std::endl;
            throw 5;
        }

        // Read the values from the file
        std::string line;
        if (std::getline(rfile, line)) {
            imgWidth = std::stoi(line);
        }
        if (std::getline(rfile, line)) {
            imgHeight = std::stoi(line);
        }
    }

    unsigned int numMCUsWidth = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;

    unsigned int i = 0; 
    for (unsigned int row = 0; row < numMCUsHeight; row++) {
            for (unsigned int col = 0; col < numMCUsWidth; col++) {
                unsigned int startX = col * MCU_SIZE;
                unsigned int startY = row * MCU_SIZE;

                // Let's create a MCU from pixel in the specified area:
                MinimumCodedUnit mcu(imgWidth, imgHeight, startY, startX);
                mcu.readCompressedFromFile(outputFolderPath, i);

                // Add the new unit to MCUs' vector of image:
                imageMCUs.push_back(mcu);
                i++;
            }
        }     
    hasFreqValues = true;

    std::cout << "**** Finished Reading Compressed ****" << std::endl;

}

void Image::writeCompressed() {

    if (!hasFreqValues){
        std::cerr << "There are not frequency values to write here!" << std::endl;
        throw 2;
    }

    std::string outputFolderPath = encodedFolderPath + imgName;

    // If directory doesn't exist, create it: 
    if (!std::filesystem::exists(outputFolderPath)) {
        std::filesystem::create_directory(outputFolderPath);
    }

    // Write a file with the metadata of the image:
    {
        std::string outFile = outputFolderPath + "/metadata.txt";
        std::ofstream wfile(outFile);

        wfile << std::to_string(imgWidth) << std::endl;
        wfile << std::to_string(imgHeight) << std::endl;
        wfile << std::to_string(imageMCUs.size()) << std::endl;
    }
    std::cout << "**** Starting to write all the MCUs ****" << std::endl;
    std::cout << " \tCompleted: 0 / " << imageMCUs.size();
    for ( unsigned int i= 0; i< imageMCUs.size(); ++i){
        imageMCUs[i].writeCompressedOnFile(outputFolderPath, i);
        std::cout << " \r \tCompleted: " << i+1 << " / " << imageMCUs.size();
    }
    std::cout << std::endl;
}

void Image::writeImage(){

    if (!hasPixelsValues){
        std::cerr << "There are not pixel values to write here!" << std::endl;
        throw 2;
    }

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

            mcu.writeImage(&imageBuffer[(startY * imgWidth + startX) * NUM_CHANNELS]);

        }
    }

    // stb_image_write to write finale JPEG image:
    stbi_write_jpg( (jpegImgsFolderPath + imgName + "restored.jpg").c_str(), imgWidth, imgHeight, NUM_CHANNELS, imageBuffer.data(), QUALITY);
}
