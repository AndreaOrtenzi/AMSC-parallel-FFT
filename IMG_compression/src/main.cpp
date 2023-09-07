#include "../inc/Image.hpp"

#include <string>

int main(){

    // Specify folders paths and image name:
    std::string jpegImgsFolderPath = "./imgs/";
    std::string encodedFolderPath = "./imgs/compressed/";
    std::string imgName = "SpongebobJPG";

    // Create a new istance of Image and read the input file image:
    Image image(jpegImgsFolderPath, encodedFolderPath, imgName, false);

    // Apply FFT:
    image.trasform();

    // Write compressed image:
    image.writeCompressed();

    // Apply Inverse transform:
    image.iTrasform();

    // Read compressed image:
    Image newImage(jpegImgsFolderPath, encodedFolderPath, imgName, true);

    // Apply iFFT:
    newImage.iTrasform();

    // New encoded image:
    newImage.writeImage();

    return 0;
}
