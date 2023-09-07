#include "../inc/Image.hpp"

#include <string>

int main(){

    // Specify folders paths and image name:
    std::string jpegImgsFolderPath = "./imgs/";
    std::string encodedFolderPath = "./imgs/compressed/";
    std::string imgName = "SpongebobJPGGray";

    // Create a new istance of Image and read the input file image:
    Image image(jpegImgsFolderPath, encodedFolderPath, imgName, false);

    // Apply FFT:
    image.transform();

    // Write compressed image:
    image.writeCompressed();

    // Apply Inverse transform:
    image.iTransform();

    // Read compressed image:
    Image newImage(jpegImgsFolderPath, encodedFolderPath, imgName, true);

    // Apply iFFT:
    newImage.iTransform();

    // New encoded image:
    newImage.writeImage();

    return 0;
}
