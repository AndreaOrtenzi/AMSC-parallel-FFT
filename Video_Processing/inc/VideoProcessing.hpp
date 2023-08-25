#include "../../DCT2D/inc/DCT_2D.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include "ffmpeg.h" // libreria per l'estrazione dei frame da un video .mp4
#include "stb_image.h" // libreria per la conversione di un frame in una matrice di interi
#include "stb_image_write.h" // libreria per salvare i nuovi frame in .jpg
#include "Eigen/Dense" 
#include "Eigen/Sparse" 
#include "Eigen/Core"
#include "unsupported/Eigen/SparseExtra"

using namespace std;
using namespace Eigen;
using Mat = Eigen::MatrixXi; // integer entries for matrix, we need it for storing matrices from frames

class VideoProcessing {
public:
    VideoProcessing(const std::string& videoFilePath);
    void ProcessVideo();
    
private:
    std::string videoFilePath;
    DCT_2D dct; 
    Mat Q;
    int frame_rows;
    int frame_cols;
    
    void ExtractFrames(std::vector<Mat>& frames);
    void DivideIntoBlocks(const Mat& frame, std::vector<Mat>& blocks);
    void Subtract128(std::vector<Mat>& blocks);
    void ApplyDCT(const std::vector<Mat>& blocks, std::vector<Mat>& frequency_blocks);
    void Quantization(std::vector<Mat>& frequency_blocks);
    void DecodingBlocks(std::vector<Mat>& frequency_blocks);
    void InverseDCT(std::vector<Mat>& frequency_blocks);
    void ReconstructFrame(const std::vector<Mat>& processedBlocks, Mat& reconstructedFrame);
    void SaveVideo(const std::string& outputVideoFilePath, const std::vector<Mat>& frames);
};
