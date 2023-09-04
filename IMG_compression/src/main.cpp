#include "../inc/MinimumCodedUnit.hpp"
#include "Eigen/Dense"


int main(){

    // const int numMatrices = 1; // Change this to the desired number of matrices
    // Eigen::Matrix<double, MCU_SIZE, MCU_SIZE> matrixArray[numMatrices];

    // // Initialize each matrix
    // for (int i = 0; i < numMatrices; ++i) {
    //     // You can populate each matrix 'matrixArray[i]' here as needed.

    //     // For example, to change the value at (3,1) in the matrix at index 1 to 71:
    //     if (i == 1) {
    //         matrixArray[i](3, 1) = 71.0;
    //     }
    // }
    MinimumCodedUnit img;
    img.transform();

    img.writeCompressedOnFile("./imgs/compressed/test",0);

    img.iTransform();


    return 0;
}