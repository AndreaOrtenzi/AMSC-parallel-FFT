#include <string.h>
#include <vector>
#include "MinimumCodedUnit.hpp"

#include "../../lib/stb_image.h"

class Image {
public:
    // open file and split pixels into minimum coded units
    Image(std::string inputFilePath, std::string outputFilePath, bool isInputCompressed = false);

    void trasform();
    void iTrasform();

    void writeCompressed();
    void writeImage();
    

private:
    void readCompressed();
    void readImage();


    void divideIntoBlocks(const Mat& frame, std::vector<Mat>& blocks);

    std::vector<MinimumCodedUnit> imageMCUs;


};