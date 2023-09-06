#include "../inc/MinimumCodedUnit.hpp"
    

void MinimumCodedUnit::readImage(unsigned char* bufferPointer){
    
    // read data and save them in mcuValues matrix
    for (unsigned int r = 0; r < dataHeight; ++r) {

        for (unsigned int c = 0; c < dataWidth; ++c){
            
            // it should be unrolled
            for (unsigned int j = 0; j < NUM_CHANNELS; ++j){
                mcuValues[j][r][c] = bufferPointer[imgWidth*r*NUM_CHANNELS + c*NUM_CHANNELS + j];
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

    havePixelsValues = true;
}


void MinimumCodedUnit::transform(){

    if (!havePixelsValues){
        std::cerr << "There are not pixels values in here!" << std::endl;
        throw 1;
    }

    // subtract 128
    int *p = &mcuValues[0][0][0];
    for (unsigned int i = 0; i < NUM_CHANNELS*MCU_SIZE*MCU_SIZE; ++i)
        p[i]-=128;

    
    // Apply FFT2D and quantizate the norm by Q
    FFT2DwithQuantization();

    haveFreqValues = true;
}
constexpr unsigned numberOfBits(unsigned x) {
    return x < 2 ? x : 1+numberOfBits(x >> 1);
}

void MinimumCodedUnit::iTransform(){

    if (!haveFreqValues){
        std::cerr << "There are not frequency values in here!" << std::endl;
        throw 2;
    }

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
        constexpr unsigned int numBits = numberOfBits(MCU_SIZE) - 1;
        // Coefficient necessary for inverse FFT:
        constexpr double N_inv = 1.0 / static_cast<double>(MCU_SIZE * MCU_SIZE);
        
        //First pass: Apply iFFT to each row:
        for (unsigned int i = 0; i < MCU_SIZE; ++i) {
            std::vector<std::complex<double>> row_vector(MCU_SIZE); 
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
                        std::complex<double> t = w * std::polar(static_cast<double>(normFreqDense[channel][i][k + j + m / 2]), static_cast<double>(phaseFreqDense[channel][i][k + j + m / 2])); 
                        std::complex<double> u = std::polar(static_cast<double>(normFreqDense[channel][i][k + j]), static_cast<double>(phaseFreqDense[channel][i][k + j]));
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
                constexpr unsigned int m = 1U << numBits; 
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
                        // Round to the nearest integer, add 128 and allocate the value to the element of the mcu values restored:
                        mcuValuesRestored[channel][k + j][i] = static_cast<int>((u + t).real() * N_inv + 128 + 0.5);
                        mcuValuesRestored[channel][k + j + m / 2][i] = static_cast<int>((u - t).real() * N_inv + 128 + 0.5);
                        w *= wm;
                    }
                } 
             
            }
        }
    }
    
    havePixelsValues = true;
    
}

void MinimumCodedUnit::writeCompressedOnFile(std::string &outputFolder, int mcuIdx){

    if (!haveFreqValues){
        std::cerr << "There are not frequency values to write!" << std::endl;
        throw 2;
    }

    // // Creates the file name for the phase matrix and the norm matrix:
    // std::string matricesFilename = outputFolder + "/mcu_" + std::to_string(mcuIdx) + "_channel_";

    // for (unsigned int channel = 0; channel < NUM_CHANNELS; ++channel) {

    //     // Use eigen to write matrices
    //     Eigen::Matrix<double, MCU_SIZE, MCU_SIZE> phaseFreqDenseEigen;
    //     Eigen::SparseMatrix<int> normFreqSparseEigen(MCU_SIZE, MCU_SIZE);

    //     // Copy from eigen to static matrices, fill normFreq and phaseFreq Eigen matrices:
    //     for(unsigned int i = 0; i < MCU_SIZE; i++){ 
    //         for(unsigned int j = 0; j < MCU_SIZE; j++){
    //             normFreqSparseEigen.coeffRef(i, j) = normFreqDense[channel][i][j];
    //             phaseFreqDenseEigen(i, j) = phaseFreqDense[channel][i][j];
    //         }
    //     }

    //     // Save norm compressed matrix:
    //     Eigen::saveMarket(normFreqSparseEigen, matricesFilename + std::to_string(channel) + "_norm.mtx");

    //     // Save phase compressed matrix:
    //     Eigen::saveMarket(phaseFreqDenseEigen, matricesFilename + std::to_string(channel) + "_phase.mtx");
        
    // }  
}

void MinimumCodedUnit::readCompressedFromFile(std::string &inputFolder, int mcuIdx){

    // std::string matricesFilename = inputFolder + "/mcu_" + std::to_string(mcuIdx) + "_channel_";

    // for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++) {
        
    //     // Use eigen to read matrices, need phaseFreq in sparse version:
    //     Eigen::SparseMatrix<double> phaseFreqEigen(MCU_SIZE, MCU_SIZE);
    //     Eigen::SparseMatrix<int> normFreqEigen(MCU_SIZE, MCU_SIZE); 

    //     // Read norm compressed matrix:
    //     Eigen::loadMarket(normFreqEigen, matricesFilename + std::to_string(channel) + "_norm.mtx");

    //     // Read phase compressed matrix:
    //     Eigen::loadMarket(phaseFreqEigen, matricesFilename + std::to_string(channel) + "_phase.mtx");


    //     // Copy from eigen to static matrices and fill mcuValuesRestored Eigen matrix: 
        
    //     for(unsigned int i = 0; i < MCU_SIZE; i++){ 
    //         for(unsigned int j = 0; j < MCU_SIZE; j++){
    //             normFreqDense[channel][i][j]= normFreqEigen.coeffRef(i,j);
    //             phaseFreqDense[channel][i][j] = phaseFreqEigen.coeffRef(i,j);
    //         }
    //     }
        
    // }

    // haveFreqValues = true;
    
}

template <typename RT,typename PT>
inline RT my_cast(PT val){
    if (std::is_floating_point<RT>::value) {
        return static_cast<RT> (val);
    }
    return static_cast<RT> (val+0.5);
}

void MinimumCodedUnit::FFT2DwithQuantization(){
        
    // &mcuValues[w][0][0],&normFreq[w][0][0],&phaseFreq[w][0][0]
    constexpr unsigned int numBits = numberOfBits(MCU_SIZE) - 1;
    // Apply for each channel:
    for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++) {
        
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

                        normFreqDense[channel][k + j][i] = my_cast<norm_type>(std::abs(u + t) / Q[k + j][i]);
                        phaseFreqDense[channel][k + j][i] = my_cast<phase_type>(std::arg(u + t));
                        
                        

                        normFreqDense[channel][k + j + m / 2][i] = my_cast<norm_type>(std::abs(u - t) / Q[k + j + m / 2][i]);
                        phaseFreqDense[channel][k + j + m / 2][i] = my_cast<phase_type>(std::arg(u - t));
                        w *= wm;
                        
                    }
                }
            } 
        }
    } // for channels 
}

void MinimumCodedUnit::writeImage(unsigned char* bufferPointer){

    if (!havePixelsValues){
        std::cerr << "There are not pixels to write!" << std::endl;
        throw 1;
    }
    for (unsigned int i = 0; i < dataHeight; ++i)
    {
        for (unsigned int j = 0; j < dataWidth; ++j)
        {
            for (unsigned int channel = 0; channel < NUM_CHANNELS; channel++)
            {
                int pixelValue = mcuValuesRestored[channel][i][j]; // TODO delete restored, use only one matrix
                bufferPointer[imgWidth * i * NUM_CHANNELS + j * NUM_CHANNELS + channel] = static_cast<unsigned char>(pixelValue);
            }
        }
    }
}

void MinimumCodedUnit::addToCompressClass(Compression<norm_type> &comp_norm, Compression<phase_type> &comp_phase){
    if (!haveFreqValues){
        std::cerr << "There are not frequency values to compress!" << std::endl;
        throw 2;
    }

    auto *p_norm = &normFreqDense[0][0][0];
    auto *p_phase = &phaseFreqDense[0][0][0];

    for (unsigned int i = 0; i < NUM_CHANNELS*MCU_SIZE*MCU_SIZE; ++i)
    {
        comp_norm.add(p_norm[i]);
        comp_phase.add(p_phase[i]);
    }
    
}

void MinimumCodedUnit::getFromCompressClass(Compression<norm_type>::Iterator &comp_norm, Compression<phase_type>::Iterator &comp_phase){
    auto *p_norm = &normFreqDense[0][0][0];
    auto *p_phase = &phaseFreqDense[0][0][0];

    for (unsigned int i = 0; i < NUM_CHANNELS*MCU_SIZE*MCU_SIZE; ++i)
    {

        p_norm[i] = *comp_norm;
        comp_norm++;
        p_phase[i] = *comp_phase;
        comp_phase++;
    }

    haveFreqValues = true;
}