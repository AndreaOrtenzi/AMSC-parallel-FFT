#include "../inc/Image.hpp"
#include <string>


int main(){
    // TESTING MCU class with constructor commented: 
    // MinimumCodedUnit img;
    // img.transform();

    // std::string path = "./imgs/compressed/test"; 
    // img.writeCompressedOnFile(path, 0);
    // img.readCompressedFromFile(path, 0);
    // img.iTransform();

    // img.printRestored();

    // TESTING Image:

        // Specify folders paths and image name:
        std::string jpegImgsFolderPath = "./imgs/";
        std::string encodedFolderPath = "./imgs/compressed/";
        std::string imgName = "SpongebobJPG";

        Image image(jpegImgsFolderPath, encodedFolderPath, imgName, false);

        // Read image and transform:
        image.trasform();

        // write compressed image:
        image.writeCompressed();

        // inverse transf:
        image.iTrasform();

        // new encoded image:
        image.writeImage();


    return 0;
}
