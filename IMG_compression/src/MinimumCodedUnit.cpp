#include "../inc/MinimumCodedUnit.hpp"

MinimumCodedUnit::MinimumCodedUnit(unsigned char* initialSquare, const unsigned int width, const unsigned int height, const unsigned int rowIdx, const unsigned int colIdx)
    : dataWidth( width-colIdx < MCU_SIZE ? width-colIdx : MCU_SIZE )
    , dataHeight( height-rowIdx < MCU_SIZE ? height-rowIdx : MCU_SIZE )
    , imgWidth(width)
    , imgHeight(height) {
    
    // Q_eig << 16, 11, 10, 16, 24, 40, 51, 61,
    //           12, 12, 14, 19, 26, 58, 60, 55,
    //           14, 13, 16, 24, 40, 57, 69, 56,
    //           14, 17, 22, 29, 51, 87, 80, 62,
    //           18, 22, 37, 56, 68, 109, 103, 77,
    //           24, 35, 55, 64, 81, 104, 113, 92,
    //           49, 64, 78, 87, 103, 121, 120, 101,
    //           72, 92, 95, 98, 112, 100, 103, 99;


    double pow_MCU_SIZE = log2(MCU_SIZE);

    // read data and save them in mcuValues matrix
    for (unsigned int r = 0; r < dataHeight; ++r) {

        for (unsigned int c = 0; c < dataWidth; ++c){
            
            // it should be unrolled
            for (unsigned int j = 0; j < NUM_CHANNELS; ++j){
                mcuValues[j][r][c] = initialSquare[width*r*NUM_CHANNELS + c*NUM_CHANNELS + j];
            }
        }
        // add padding if needed
        for(unsigned int c = dataWidth; c < MCU_SIZE; ++c) {
            for (unsigned int j = 0; j < NUM_CHANNELS; ++j){
                mcuValues[j][r][c] = mcuValues[j][r][c-1];
            }
        }
    }
    for (unsigned int r = dataHeight; r < MCU_SIZE; ++r) {        
        for (unsigned int j = 0; j < NUM_CHANNELS; ++j){
            std::copy(&mcuValues[j][r-1][0],&mcuValues[j][r-1][MCU_SIZE],&mcuValues[j][r][0]);
        }
    }
    
}


void MinimumCodedUnit::transform(){

    // subtract 128
    for (unsigned int w = 0; w < NUM_CHANNELS; ++w)
        for (unsigned int i = 0; i < MCU_SIZE; ++i)
            for (unsigned int j = 0; j < MCU_SIZE; ++j) 
                mcuValues[w][i][j] -= 128;

    
    // Apply FFT2D and quantizate the norm by Q
    FFT2DwithQuantization();

    // Fill normFreq and phaseFreq Eigen matrices:
     for(unsigned int channel=0; channel<NUM_CHANNELS; channel++){
        for(unsigned int i = 0; i < MCU_SIZE; i++){ 
            for(unsigned int j = 0; j < MCU_SIZE; j++){
                normFreqDenseEigen[channel](i, j) = normFreqDense[channel][i][j];
                phaseFreqDenseEigen[channel](i, j) = phaseFreqDense[channel][i][j];
            }
        }
    }


}
constexpr unsigned numberOfBits(unsigned x) {
    return x < 2 ? x : 1+numberOfBits(x >> 1);
}

void MinimumCodedUnit::iTransform(){
    std::cout << "#############################################################################" << std::endl;
    std::cout << "Starting decompression phase:" << std::endl;
    // Step 1: multiply element wise per Q matrix: 
    for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++){
        for (unsigned int i = 0; i < MCU_SIZE; ++i){
            for (unsigned int j = 0; j < MCU_SIZE; ++j){
                normFreqDense[channel][i][j] *=  Q[i][j];
            }
        }
    }

    

    // Step 2: Take the 2-dimensional inverse FFT:
    for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++){
        std::complex<double> input_cols[MCU_SIZE][MCU_SIZE];

        // Num bits:
        unsigned int numBits = static_cast<unsigned int>(log2(MCU_SIZE));
        // Coefficient necessary for inverse FFT:
        constexpr double N_inv = 1.0 / static_cast<double>(MCU_SIZE * MCU_SIZE);
        
        //First pass: Apply iFFT to each row:
        for (unsigned int i = 0; i < MCU_SIZE; ++i) {
            std::vector<complex<double>> row_vector(MCU_SIZE); 
            for (unsigned int l = 0; l < MCU_SIZE; l++) {
                unsigned int j = 0;
                for (unsigned int k = 0; k < numBits; k++) {
                    j = (j << 1) | ((l >> k) & 1U);
                }
                if (j > l) {
                    std::swap(normFreqDense[channel][i][l], normFreqDense[channel][i][j]); 
                    std::swap(phaseFreqDense[channel][i][l], phaseFreqDense[channel][i][j]);
                }
            }
            
            {
                constexpr unsigned int m = 1U << 1; 
                std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * std::polar(static_cast<double>(normFreqDense[channel][i][k + j + m / 2]), phaseFreqDense[channel][i][k + j + m / 2]); 
                        std::complex<double> u = std::polar(static_cast<double>(normFreqDense[channel][i][k + j]), phaseFreqDense[channel][i][k + j]);
                        row_vector[k + j] = u + t; 
                        row_vector[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            } 
            
            for (unsigned int s = 2; s < numBits; s++) {
                unsigned int m = 1U << s; 
                std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) { 
                        std::complex<double> t = w * row_vector[k + j + m / 2];
                        std::complex<double> u = row_vector[k + j];
                        row_vector[k + j] = u + t;
                        row_vector[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            }
            // s == numBits 
            
            {
                unsigned int m = 1U << numBits; 
                std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * row_vector[k + j + m / 2];
                        std::complex<double> u = row_vector[k + j];
                        input_cols[k + j][i] = u + t;
                        input_cols[k + j + m / 2][i] = u - t;
                        w *= wm;
                    }
                }
            }

        }
        
        //Second pass: Apply iFFT to each column
        for (unsigned int i = 0; i < MCU_SIZE; ++i) {
            std::complex<double>* col_vector = &input_cols[i][0];
            for (unsigned int l = 0; l < MCU_SIZE; l++) {
                unsigned int j = 0;
                    for (unsigned int k = 0; k < numBits; k++) {
                        j = (j << 1) | ((l >> k) & 1U);
                    }
                    if (j > l) {
                        std::swap(col_vector[l], col_vector[j]);
                    }
                } 
            for (unsigned int s = 1; s < numBits; s++) {
                unsigned int m = 1U << s; 
                std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * col_vector[k + j + m / 2];
                        std::complex<double> u = col_vector[k + j];
                        col_vector[k + j] = u + t;
                        col_vector[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            }
            // s == numBits
            {
                unsigned int m = 1U << numBits; 
                std::complex<double> wm = std::exp(2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * col_vector[k + j + m / 2];
                        std::complex<double> u = col_vector[k + j];
                        // std::cout<<"Value NOT APPROX in the indexes: [" << channel <<"],row[" << k+j <<"],col[" <<i <<"] = " << (u+t).real()*N_inv << std::endl;
                        // Round to the nearest integer, add 128 and allocate the value to the element of the mcu values restored:
                        mcuValuesRestored[channel][k + j][i] = static_cast<int>((u + t).real() * N_inv + 128 + 0.5);
                        mcuValuesRestored[channel][k + j + m / 2][i] = static_cast<int>((u - t).real() * N_inv + 128 + 0.5);
                        w *= wm;
                    }
                } 
             
            }
        }
    }
    // Fill mcuValuesRestored Eigen matrix: 
    for(unsigned int channel=0; channel<NUM_CHANNELS; channel++){
        for(unsigned int i = 0; i < MCU_SIZE; i++){ 
            for(unsigned int j = 0; j < MCU_SIZE; j++){
                mcuValuesRestoredEigen[channel](i, j) = mcuValuesRestored[channel][i][j];
            }
        }
    }

    std::cout << "Ended decompression phase." << std::endl;
    std::cout << "#############################################################################" << std::endl;
    
    
}

void MinimumCodedUnit::writeCompressedOnFile(std::string &outputFolder, int mcuIdx){

    // Creates the file name for the phase matrix and the norm matrix:
    std::string normMatrixFilename = outputFolder + "/mcu_" + std::to_string(mcuIdx) + "_norm.mtx";
    std::string phaseMatrixFilename = outputFolder + "/mcu_" + std::to_string(mcuIdx) + "_phase.mtx";

    // Save norm compressed matrix:
    Eigen::saveMarket(normFreqDenseEigen, normMatrixFilename);

    // Save phase compressed matrix:
    Eigen::saveMarket(phaseFreqDenseEigen, phaseMatrixFilename);
    
}

void MinimumCodedUnit::FFT2DwithQuantization(){
        
    // &mcuValues[w][0][0],&normFreq[w][0][0],&phaseFreq[w][0][0]
    constexpr unsigned int numBits = numberOfBits(MCU_SIZE) - 1;
    std::cout << "#############################################################################" << std::endl;
    std::cout << "Starting compression phase:" << std::endl;
    // Apply for each channel:
    for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++) {
        
        Eigen::Matrix<double, MCU_SIZE, MCU_SIZE> matrix;
        std::complex<double> input_cols[MCU_SIZE][MCU_SIZE];

        //First pass: Apply FFT to each row
        for (unsigned int i = 0; i < MCU_SIZE; ++i) {
            std::complex<double> row_vector[MCU_SIZE];
            
            for (unsigned int l = 0; l < MCU_SIZE; l++) { // **************
                unsigned int ji = 0;
                for (unsigned int k = 0; k < numBits; k++) {
                    ji = (ji << 1) | ((l >> k) & 1U);
                }
                if (ji > l) {
                    std::swap(mcuValues[channel][i][l], mcuValues[channel][i][ji]);
                }
            } 
            // use last iteration to write column vectors and the first to not override input_matrix
            // s = 1
            {
                unsigned int m = 1U << 1; 
                std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {

                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * static_cast<std::complex<double>>(mcuValues[channel][i][k + j + m / 2]);
                        std::complex<double> u = static_cast<std::complex<double>>(mcuValues[channel][i][k + j]);
                        row_vector[k + j] = u + t;
                        row_vector[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            }
            // swap again to restore original input_matrix
            
            for (unsigned int l = 0; l < MCU_SIZE; l++) { // **************
                unsigned int ji = 0;
                for (unsigned int k = 0; k < numBits; k++) {
                    ji = (ji << 1) | ((l >> k) & 1U);
                }
                if (ji > l) {
                    std::swap(mcuValues[channel][i][l], mcuValues[channel][i][ji]);
                }
            }

            for (unsigned int s = 2; s < numBits; s++) {
                unsigned int m = 1U << s; 
                std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * row_vector[k + j + m / 2];
                        std::complex<double> u = row_vector[k + j];
                        row_vector[k + j] = u + t;
                        row_vector[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            } 
            // s == numBits
            {
                constexpr unsigned int m = 1U << numBits; 
                std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * row_vector[k + j + m / 2];
                        std::complex<double> u = row_vector[k + j];
                        input_cols[k + j][i] = u + t;
                        input_cols[k + j + m / 2][i] = u - t;
                        w *= wm;
                    }
                }
            }
        } 

        //Second pass: Apply FFT to each column
        for (unsigned int i = 0; i < MCU_SIZE; ++i) {
            std::complex<double> *col_vector = &input_cols[i][0];
            
            for (unsigned int l = 0; l < MCU_SIZE; l++){
                unsigned int j = 0;
                for (unsigned int k = 0; k < numBits; k++) {
                    j = (j << 1) | ((l >> k) & 1U);
                }
                if (j > l) {
                    std::swap(col_vector[l], col_vector[j]);
                }
            }

            for (unsigned int s = 1; s < numBits; s++) {
                unsigned int m = 1U << s;
                std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * col_vector[k + j + m / 2];
                        std::complex<double> u = col_vector[k + j];
                        col_vector[k + j] = u + t;
                        col_vector[k + j + m / 2] = u - t;
                        w *= wm;
                    }
                }
            } 
            // s == numBits
            {
                constexpr unsigned int m = 1U << numBits; 
                std::complex<double> wm = std::exp(-2.0 * M_PI * std::complex<double>(0, 1) / static_cast<double>(m));
                for (unsigned int k = 0; k < MCU_SIZE; k += m) {
                    std::complex<double> w = 1.0;
                    for (unsigned int j = 0; j < m / 2; j++) {
                        std::complex<double> t = w * col_vector[k + j + m / 2];
                        std::complex<double> u = col_vector[k + j];

                        normFreqDense[channel][k + j][i] = static_cast<int>(std::abs(u + t) / Q[k + j][i] + 0.5);
                        phaseFreqDense[channel][k + j][i] = std::arg(u + t);
                        
                        

                        normFreqDense[channel][k + j + m / 2][i] = static_cast<int>(std::abs(u - t) / Q[k + j + m / 2][i] + 0.5);
                        phaseFreqDense[channel][k + j + m / 2][i] = std::arg(u - t);
                        //std::cout << "Print element in the indexes: channel[" << channel <<"],row[" << k+j <<"],col[" << i <<"] = " << static_cast<int>(std::abs(u + t) / Q[k + j][i] + 0.5) << std::endl;
                        //std::cout << "Print element in the indexes: channel[" << channel <<"],row[" << k+j+m/2 <<"],col[" << i <<"] = " << static_cast<int>(std::abs(u - t) / Q[k + j + m / 2][i] + 0.5) << std::endl;
                        w *= wm;
                        
                    }
                }
            } 
        }
    } // for channels 
    std::cout << "Ended compression phase." << std::endl;
    std::cout << "#############################################################################" << std::endl;

}

void MinimumCodedUnit::writeImage(unsigned char* bufferPointer){
        for (unsigned int i = 0; i < dataHeight; ++i) {
            for (unsigned int j = 0; j < dataWidth; ++j) {
                for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++) {
                    int pixelValue = mcuValuesRestored[channel][i][j];
                    bufferPointer[imgWidth*i*NUM_CHANNELS + j*NUM_CHANNELS + channel] = static_cast<unsigned char>(pixelValue);
                }
            }
        }
}