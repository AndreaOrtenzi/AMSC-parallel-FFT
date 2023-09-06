#include <string.h>
#include <vector>
#include <string>

#include <cmath>
#include <algorithm> // to use std::for_each
#include <filesystem> // for create directory compressed_images
#include "parameters"

#include "MinimumCodedUnit.hpp"
// #include "../../Compression/inc/Compression.hpp"

using namespace std;

class Image {
public:
    Image(std::string jpegImgsFolderPath_, std::string encodedFolderPath_, std::string imgName_, bool isInputCompressed);

    void trasform();
    void iTrasform();

    // usa stb_image_write:
    void writeCompressed(); 
    void writeImage(); 
    

private:
    std::string jpegImgsFolderPath;
    std::string encodedFolderPath;
    std::string imgName;
    bool hasFreqValues, hasPixelsValues;

    void readCompressed();
    void readImage(); 

    int imgHeight, imgWidth;

    // Image's MCUs vector
    std::vector<MinimumCodedUnit> imageMCUs;


};