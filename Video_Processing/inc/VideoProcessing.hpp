#include "../../FFT2D/inc/FFT_2D.hpp"
#include <iostream>
#include "ffmpeg.h" // libreria per l'estrazione dei frame da un video .mp4
#include "stb_image.h" // libreria per la conversione di un frame in una matrice di interi
#include "stb_image_write.h" // libreria per salvare i nuovi frame in .jpg

using namespace std;
using namespace Eigen;
using Mat = Eigen::MatrixXcd; // integer entries for matrix, we need it for storing matrices from frames

class VideoProcessing {
public:
    VideoProcessing(const std::string& videoFilePath);
    void ProcessVideo();
    
private:
    std::string videoFilePath;
    Mat Q;
    int frame_rows;
    int frame_cols;
    int numFrames;
    
    void ExtractFrames(std::vector<Mat>& frames);
    void DivideIntoBlocks(const Mat& frame, std::vector<Mat>& blocks);
    void Subtract128(std::vector<Mat>& blocks);
    void ApplyFFT(std::vector<Mat>& blocks, std::vector<Mat>& frequency_blocks);
    void Quantization(std::vector<Mat>& frequency_blocks);
    void DecodingBlocks(std::vector<Mat>& frequency_blocks);
    void InverseFFT(std::vector<Mat>& frequency_blocks);
    void ReconstructFrames(const std::vector<Mat>& frequency_blocks, std::vector<Mat>& reconstructed_frames);
    void SaveVideo(std::vector<Mat>& reconstructed_frames);
};
