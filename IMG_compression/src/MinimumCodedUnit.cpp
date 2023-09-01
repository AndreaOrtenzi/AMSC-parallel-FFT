#include "../inc/MinimumCodedUnit.hpp"

MinimumCodedUnit::MinimumCodedUnit(unsigned char* initialSquare, const unsigned int width, const unsigned int height, const unsigned int rowIdx, const unsigned int colIdx)
    : dataWidth( width-colIdx < MCU_SIZE ? width-colIdx : MCU_SIZE )
    , dataHeight( height-rowIdx < MCU_SIZE ? height-rowIdx : MCU_SIZE )
    , imgWidth(width)
    , imgHeight(height) {
        
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


void MinimumCodedUnit::trasform(){

    // subtract 128
    for (unsigned int w = 0; w < NUM_CHANNELS; ++w)
        for (unsigned int i = 0; i < MCU_SIZE; ++i)
            for (unsigned int j = 0; j < MCU_SIZE; ++j) 
                mcuValues[w][i][j] -= 128;

    // Apply FFT2D and quantizate the norm by Q
    FFT2DwithQuantization();

    // Fill eigen matrices ?

}

constexpr unsigned numberOfBits(unsigned x) {
    return x < 2 ? x : 1+numberOfBits(x >> 1);
}

void MinimumCodedUnit::iTrasform(){
    // 2. Multiply elementwise by Q.

    // 3. Take the 2-dimensional inverse FFT.

    // 4. Round to the nearest integer, and add 128 so that the entries are between 0 and 255

}

void MinimumCodedUnit::writeCompressedOnFile(std::ofstream& writeFilePointer){

}

void MinimumCodedUnit::FFT2DwithQuantization(){
        
    // &mcuValues[w][0][0],&normFreq[w][0][0],&phaseFreq[w][0][0]
    constexpr unsigned int numBits = numberOfBits(MCU_SIZE);
    
    // Apply for each channel:
    for (unsigned int channel = 0; channel < NUM_CHANNELS; ++channel) {
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
                        
                        phaseFreq[channel][k + j][i] = std::arg(u + t);
                        normFreq[channel][k + j][i] = static_cast<int>(std::abs(u + t) / Q[k + j][i] + 0.5);

                        phaseFreq[channel][k + j + m / 2][i] = std::arg(u - t);
                        normFreq[channel][k + j + m / 2][i] = static_cast<int>(std::abs(u - t) / Q[k + j + m / 2][i] + 0.5);
                        
                        w *= wm;
                    }
                }
            }
        }
    } // for channels
}
