#include <string.h>
#include <vector>
#include <string>

#include <cmath>
#include <algorithm> // to use std::for_each
#include <filesystem> // to create directory compressed_images
#include "parameters"

#include "MinimumCodedUnit.hpp"
#include "../../lib/VectorInFiles.hpp"

using namespace std;

class Image {
public:
    // Constructor to initialize an Image object.
    // Parameters:
    //   jpegImgsFolderPath_: The folder path containing JPEG images.
    //   encodedFolderPath_: The folder path where encoded data is stored.
    //   imgName_: The name of the image or the name of the folder in the encodedFolderPath_,
    //            depending on the value of isInputCompressed.
    //   isInputCompressed: Flag indicating whether the input data is compressed. If true,
    //                      data will be retrieved from encodedFolderPath_/imgName_;
    //                      otherwise, imgName_ represents the name of the image.
    Image(std::string jpegImgsFolderPath_, std::string encodedFolderPath_, std::string imgName_, bool isInputCompressed);


    // Perform a transformation on the image.
    // The transformation involves the following steps:
    // 1. Split the image into MCU_SIZE^2 blocks.
    // 2. Subtract 128 from the pixel values of each block.
    // 3. Apply the Fast Fourier Transform (FFT) to each block.
    // 4. Divide the resulting frequency values by a quantization (Q) matrix.
    // This transformation can be reversed using the iTrasform method.
    void transform();
    void iTransform();

    // This method saves the frequency values of the image to separate files for each color channel.
    // The file structure is as follows:
    // - There are NUM_CHANNELS channels (e.g., for Red, Green, and Blue) that are treated separately.
    // - For each Minimum Coded Unit (MCU) in the image, two values are stored (phase and magnitude).
    // - The total number of files created is 2 * the number of MCUs in the image * NUM_CHANNELS.
    void writeCompressed(); 

    // Write the pixel values to jpg file using the stb_image_write library.
    void writeImage(); 

private:
    std::string jpegImgsFolderPath;   // Folder path containing JPEG images.
    std::string encodedFolderPath;    // Folder path for encoded data.
    std::string imgName;             // The name of the image.
    bool hasFreqValues;              // Flag indicating if frequency domain values are present.
    bool hasPixelsValues;            // Flag indicating if pixel values are present.

    int imgHeight;                   // Height of the image.
    int imgWidth;                    // Width of the image.

    // Vector to store the image's Minimum Coded Units (MCUs).
    std::vector<MinimumCodedUnit> imageMCUs;

    // Private function to read compressed data from encodedFolderPath/imgName
    // It's called in the constructor only if isInputCompressed == true
    void readCompressed();

    // Private function to read jpeg data using stb_image library and store
    // pixel values in different MCUs
    void readImage(); 
};
