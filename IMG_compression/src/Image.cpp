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

// Read the image data from the specified file and create Minimum Coded Units (MCUs).
// The method reads the image pixel data, calculates the number of MCUs needed to cover
// the image, and creates MCUs from the pixel data. Each MCU corresponds to a block of 
// MCU_SIZE x MCU_SIZE pixels in the image.
// Parameters:
//   None
// Throws:
//   - Throws an exception (int value 1) if an error occurs during image reading.
void Image::readImage(){
    int numChannels = NUM_CHANNELS;
    std::string imagePath = jpegImgsFolderPath + imgName + ".jpg";
    unsigned char *imageData = stbi_load(imagePath.c_str(), &imgWidth, &imgHeight, &numChannels, 0);

    if (!imageData) {
        std::cerr << "Error occurred during the image reading." << std::endl;
        throw 1;
    }
    std::cout << "**** Starting Reading Image  ****" << std::endl;
    
    // Calculate the number of MCUs in width and height:
    unsigned int numMCUsWidth = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;

    // Create MCUs from the image's pixel data:
    for (unsigned int row = 0; row < numMCUsHeight; row++) {
        for (unsigned int col = 0; col < numMCUsWidth; col++) {
            unsigned int startX = col * MCU_SIZE;
            unsigned int startY = row * MCU_SIZE;

            // Create an MCU from the pixels in the specified area:
            MinimumCodedUnit mcu(imgWidth, imgHeight, startY, startX);
            mcu.readImage(&imageData[(startY * imgWidth + startX) * numChannels]);

            // Add the new MCU to the image's vector of MCUs:
            imageMCUs.push_back(mcu);
        }
    }

    // Free the memory used for the image data: 
    stbi_image_free(imageData);

    std::cout << "**** Read Image Finished ****" << std::endl;

    hasPixelsValues = true;
}


// Apply the FFT2D transformation to each Minimum Coded Unit (MCU) in the image.
// This method iterates through the vector of MCUs and applies the FFT2D transformation
// to each MCU using a lambda function. Parallel execution is enabled for better efficiency.
// Throws:
//   - Throws an exception (int value 1) if there are not pixel values to transform.
void Image::transform(){

    std::cout << "**** Starting FFT ****" << std::endl;

    // Check if pixel values are available in the MCUs:
    if (!hasPixelsValues){
        std::cerr << "There are no pixel values available." << std::endl;
        throw 1;
    }

    // Define a lambda function to apply transformation for each MCU:
    auto apply_FFT2D = [](MinimumCodedUnit &mcu) {
        mcu.transform(); // Call the transform method of MinimumCodeUnit
    };

    // Use std::execution::par from c++20 to parallelize the code
    std::for_each(std::execution::par, imageMCUs.begin(), imageMCUs.end(), apply_FFT2D);

    // Set the flag to indicate that frequency values are now available:
    hasFreqValues = true;

    std::cout << "**** Finished FFT ****" << std::endl;
}



// Inverse Transform every MCU in the vector imageMCUs:
// Throws:
//   - Throws an exception (int value 2) if there are not frequency values to iTransform.
void Image::iTransform() {
    if (!hasFreqValues) {
        std::cerr << "There are no frequency values available." << std::endl;
        throw 2;
    }

    std::cout << "**** Starting iFFT ****" << std::endl;

    // Define a lambda function to apply the iFFT2D for each MCU:
    auto apply_iFFT2D = [](MinimumCodedUnit &mcu) {
        mcu.iTransform(); // Call the iTransform method of MinimumCodedUnit to perform the inverse FFT.
    };

    // For each MCU, apply the lambda function defined above:
    std::for_each(std::execution::par, imageMCUs.begin(), imageMCUs.end(), apply_iFFT2D);

    hasPixelsValues = true;

    std::cout << "**** iFFT finished ****" << std::endl;
}

// Private function to read compressed data from encodedFolderPath/imgName
// It's called in the constructor only if isInputCompressed == true
// Throws:
//   - Throws an exception (int value 5) if an error occurs during metadata file reading.
void Image::readCompressed(){

    std::string outputFolderPath = encodedFolderPath + imgName;
    std::cout << "**** Starting Reading Compressed ****" << std::endl;

    // Read imgWidth and imgHeight from the metadata file
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

    unsigned int numMCUsWidth  = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCU        = numMCUsWidth * numMCUsHeight;

    std::cout << " \tCompleted: 0 / " << numMCU;
    unsigned int i = 0; 

    for (unsigned int row = 0; row < numMCUsHeight; row++) {
            for (unsigned int col = 0; col < numMCUsWidth; col++) {
                unsigned int startX = col * MCU_SIZE;
                unsigned int startY = row * MCU_SIZE;

                // Create an mcu that will read the files numbered with i
                MinimumCodedUnit mcu(imgWidth, imgHeight, startY, startX);
                mcu.readCompressedFromFile(outputFolderPath, i);

                std::cout << " \r \tCompleted: " << i + 1 << " / " << numMCU;

                imageMCUs.push_back(mcu);
                i++;
            }
        }     
    hasFreqValues = true;
    std::cout << std::endl;
    std::cout << "**** Finished Reading Compressed ****" << std::endl;

}

// Write the frequency values of the image to compressed files.
// Throws:
//   - Throws an exception (int value 2) if there are not frequency values to iTransform.
void Image::writeCompressed() {
    // Check if frequency values are available for writing.
    if (!hasFreqValues) {
        std::cerr << "There are no frequency values to write here!" << std::endl;
        throw 2; // Throw an exception to indicate the absence of frequency values.
    }

    // Define the output folder path for storing compressed data.
    std::string outputFolderPath = encodedFolderPath + imgName;

    // If the output directory doesn't exist, create it.
    if (!std::filesystem::exists(outputFolderPath)) {
        std::filesystem::create_directory(outputFolderPath);
    }

    // Write image width, height, and the number of MCUs to the metadata file.
    {
        std::string outFile = outputFolderPath + "/metadata.txt";
        std::ofstream wfile(outFile);

        wfile << std::to_string(imgWidth) << std::endl;
        wfile << std::to_string(imgHeight) << std::endl;
        wfile << std::to_string(imageMCUs.size()) << std::endl;
    }

    std::cout << "**** Starting to write all the MCUs ****" << std::endl;
    std::cout << " \tCompleted: 0 / " << imageMCUs.size();

    // Iterate through each MCU and call writeCompressedOnFile to save their frequency values to files.
    for (unsigned int i = 0; i < imageMCUs.size(); ++i) {
        imageMCUs[i].writeCompressedOnFile(outputFolderPath, i);

        std::cout << " \r \tCompleted: " << i + 1 << " / " << imageMCUs.size();
    }

    std::cout << std::endl;
}


// Write the restored image using pixel values from imageMCUs to a JPEG file.
// Throws:
//   - Throws an exception (int value 2) if there are not frequency values to iTransform.
void Image::writeImage() {
    // Check if pixel values are available
    if (!hasPixelsValues) {
        std::cerr << "There are no pixel values to write here!" << std::endl;
        throw 2;
    }

    // Create a byte array to store the final image data
    std::vector<unsigned char> imageBuffer(imgWidth * imgHeight * NUM_CHANNELS);

    // Calculate the number of MCUs in width and height
    unsigned int numMCUsWidth = (imgWidth + MCU_SIZE - 1) / MCU_SIZE;
    unsigned int numMCUsHeight = (imgHeight + MCU_SIZE - 1) / MCU_SIZE;

    // Iterate through MCUs and assemble the pixel data into the imageBuffer
    for (unsigned int row = 0; row < numMCUsHeight; row++) {
        for (unsigned int col = 0; col < numMCUsWidth; col++) {
            // Calculate the starting position of the MCU
            unsigned int startX = col * MCU_SIZE;
            unsigned int startY = row * MCU_SIZE;
            unsigned int mcuIdx = row * (imgWidth / MCU_SIZE) + col;

            MinimumCodedUnit& mcu = imageMCUs[mcuIdx];

            // Write the pixel data of the MCU to the imageBuffer
            mcu.writeImage(&imageBuffer[(startY * imgWidth + startX) * NUM_CHANNELS]);
        }
    }

    // Use stb_image_write to save the final JPEG image
    stbi_write_jpg((jpegImgsFolderPath + imgName + "restored.jpg").c_str(), imgWidth, imgHeight, NUM_CHANNELS, imageBuffer.data(), QUALITY);
}

