#include "../inc/VideoProcessing.hpp"
#define STB_IMAGE_IMPL
#include <stb_image.h>
#define STB_WRITE_IMAGE_IMPL
#include <stb_image_write.h>


void VideoProcessing::ExtractFrames(std::vector<int_Mat>& frames) {

    // Open the video utilising ffmpeg:
    AVFormatContext* formatContext = avformat_alloc_context();
    if (avformat_open_input(&formatContext, videoFilePath.c_str(), nullptr, nullptr) != 0) {
        return;
    }

    // Frame structure:
    AVFrame* frame = av_frame_alloc();
    int frameNumber = 0; 

    // Extract frame from video until av_read_frame gives a negative value, it means that all frames have been read:
    while (av_read_frame(formatContext, frame) >= 0) {
        // utilizing stb_image.h takes frame in a integer matrix
        int width = frame->width;
        int height = frame->height;
        int_Mat matrix(height, width);

        // Add matrix to frames vector:
        frames.push_back(matrix);

        // Save frame in version .jpg in 'frames' directory:
        std::string frameFileName = "../frames/frame_" + std::to_string(frameNumber++) + ".jpg";
        stbi_write_jpg(frameFileName.c_str(), width, height, 3, frame->data[0], 100);

        // Free the frame:
        av_frame_unref(frame);
    }

    // Free resources:
    avformat_close_input(&formatContext);
    av_frame_free(&frame);
}


void VideoProcessing::DivideIntoBlocks(const std::vector<int_Mat>& frames, std::vector<int_Mat>& blocks) {
    for (const int_Mat& frame : frames) {
        int numRows = frame.rows();
        int numCols = frame.cols();

        // Frame dimensions must be multiple of 8: 
        int numBlockRows = numRows / 8;
        int numBlockCols = numCols / 8;

        // Subdivide the frame in blocks:
        for (int i = 0; i < numBlockRows; ++i) {
            for (int j = 0; j < numBlockCols; ++j) {
                //Eigen::Block<Derived, Rows, Cols> MatrixType::block(Index startRow, Index startCol, Index numRows, Index numCols);
                int_Mat block = frame.block(i * 8, j * 8, 8, 8);
                blocks.push_back(block);
            }
        }
    }
}


void VideoProcessing::Subtract128(std::vector<int_Mat>& blocks) {
    for (int i = 0; i < blocks.size(); i++) {
        int_Mat& block = blocks[i];
        int numRows = block.rows();
        int numCols = block.cols();

        for (int j = 0; j < numRows; j++) {
            for (int k = 0; k < numCols; k++) {
                block(j, k) -= 128;
            }
        }
    }
}

void ConvertBlocks(const std::vector<int_Mat>& blocks, std::vector<cd_Mat>& cd_blocks) {
    cd_blocks.clear(); // Assicurati che il vettore di output sia vuoto

    for (const int_Mat& intBlock : blocks) {
        // Create new block MatrixXcd with same dimensions:
        cd_Mat cdBlock(intBlock.rows(), intBlock.cols());

        // Copies values from intBlock to cdBlock, converting from int to double:
        for (int i = 0; i < intBlock.rows(); ++i) {
            for (int j = 0; j < intBlock.cols(); ++j) {
                cdBlock(i, j) = static_cast<double>(intBlock(i, j));
            }
        }
          // Copies values from intBlock to cdBlock, converting from int to double and allocate in the real part of cdBlock:
        cdBlock.real() = intBlock.cast<double>();

        // Imaginary part of cdBlock set to zero:
        cdBlock.imag().setZero();

        // Add new block to vector
        cd_blocks.push_back(cdBlock);
    }
}



void ApplyFFT(const std::vector<cd_Mat>& input_blocks, std::vector<cd_Mat>& frequency_blocks) {
    // Assicurati che il vettore di output sia vuoto
    frequency_blocks.clear();

    for (const cd_Mat& input_block : input_blocks) {
        // Create new block with same dimensions
        cd_Mat output_block = input_block;

        // FFT on input block:
        FFT_2D fft2d(input_block, output_block);
        fft2d.transform_par();

        // Add new block to vector of complex double entries matrices
        frequency_blocks.push_back(output_block);
    }
}



void VideoProcessing::QuantizeAndRound(cd_Mat& block) {
    // 1) quantizzazione elemento per elemento utilizzando la matrice di quantizzazione Q
    // 2) arrrotondare con ceil ciascun elemento quantizzato al valore intero più vicino come nel project_summer
}

void VideoProcessing::InverseFFT(cd_Mat& block) {
    // iFFT ad ogni blocco
}

void VideoProcessing::ReconstructFrame(const std::vector<cd_Mat>& processedBlocks, int_Mat& reconstructedFrame) {
    // qua bisogna ricostruire il frame a partire dai blocchi processati
}

void VideoProcessing::SaveVideo(const std::string& outputVideoFilePath, const std::vector<int_Mat>& frames) {
    // salvataggio dei frame come video utilizzando stb_image_write.h
}
