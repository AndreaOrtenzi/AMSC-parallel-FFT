#include <string.h>
#include <vector>
#include <string>
#include "MinimumCodedUnit.hpp"

class Image {
public:
    Image(std::string inputFilePath, std::string outputFilePath, bool isInputCompressed = false);

    void trasform();
    void iTrasform();

    // usa stb_image_write:
    void writeCompressed(); 
    void writeImage(); // scrivere il file per l'immagine 
    

private:
    std::string inputFilePath;
    std::string outputFilePath;
    void readCompressed(); // legge l'immagine già compressa
    void readImage(); // legge l'immagine da comprimere


    //void divideIntoBlocks(const Mat& frame, std::vector<Mat>& blocks); --> in readImage l'ho fatto

    // Image's MCUs vector
    std::vector<MinimumCodedUnit> imageMCUs;


};