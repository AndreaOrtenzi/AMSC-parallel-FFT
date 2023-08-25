#include "../inc/DCT_2D.hpp"
#include <iostream>

int main() {
    // matrice per testing a caso: 
    Mat spatialMatrix(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            spatialMatrix(i, j) = i + j;
        }
    }

    std::cout << "Input space Matrix:" << std::endl;
    std::cout << spatialMatrix << std::endl;

    DCT_2D dct(spatialMatrix, spatialMatrix);

    dct.transform_seq();
    
    std::cout << "Frequencies matrix (seq):" << std::endl;
    std::cout << dct.getFrequencyValues() << std::endl;

    dct.transform_par();

    std::cout << "Frequencies matrix (parall):" << std::endl;
    std::cout << dct.getFrequencyValues() << std::endl;

    dct.iTransform();

    std::cout << "Inverse Transform Matrix:" << std::endl;
    std::cout << dct.getSpatialValues() << std::endl;

    return 0;
}
