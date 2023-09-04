#include <string.h>
#include <vector>
#include <string>

#include <cmath>
#include <algorithm> // to use std::for_each
#include <filesystem> // for create directory compressed_images
#include "parameters"

#include "MinimumCodedUnit.hpp"

using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#include "../../lib/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../lib/stb_image_write.h"


class Image {
public:
    Image(std::string inputFilePath, std::string outputFilePath, bool isInputCompressed = false);

    void trasform();
    void iTrasform();

    // usa stb_image_write:
    void writeCompressed(); 
    void writeImage(); // scrivere il file per l'immagine 
    

private:
    std::string jpegImgsFolderPath;
    std::string encodedFolderPath;
    std::string imgName;
    bool hasFreqValues, hasPixelsValues;

    void readCompressed(); // legge l'immagine già compressa
    void readImage(); // legge l'immagine da comprimere

    int imgHeight, imgWidth;

    // Image's MCUs vector
    std::vector<MinimumCodedUnit> imageMCUs;


};