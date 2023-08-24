#include "../../FFT2D/inc/FFT_2D.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include "ffmpeg.h" // libreria per l'estrazione dei frame da un video .mp4
#include "stb_image.h" // libreria per la conversione di un frame in una matrice di interi
#include "stb_image_write.h" // libreria per salvare i nuovi frame in .jpg
#include "Eigen/Dense" 
#include "Eigen/Sparse" 
#include "Eigen/Core"
#include "unsupported/Eigen/SparseExtra"

using namespace std;
using namespace Eigen;
using int_Mat = Eigen::MatrixXi; // integer entries for matrix, we need it for storing matrices from frames
using cd_Mat = Eigen::MatrixXcd; // complex double entries for matrix, we need it for FFT_2D class


class VideoProcessing {
public:
    VideoProcessing(const std::string& videoFilePath);
    void ProcessVideo();
    
private:
    std::string videoFilePath;
    FFT_2D fft; 
    
    void ExtractFrames(std::vector<intMat>& frames);
    void DivideIntoBlocks(const intMat& frame, std::vector<intMat>& blocks);
    void Subtract128(intMat& block);
    void ConvertBlocks(const std::vector<int_Mat>& blocks, std::vector<cd_Mat>& cd_blocks)
    void ApplyFFT(doubleMat& block);
    void QuantizeAndRound(doubleMat& block);
    void InverseFFT(doubleMat& block);
    void ReconstructFrame(const std::vector<doubleMat>& processedBlocks, intMat& reconstructedFrame);
    void SaveVideo(const std::string& outputVideoFilePath, const std::vector<intMat>& frames);
};
