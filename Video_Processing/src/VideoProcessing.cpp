#include "../inc/VideoProcessing.hpp"
#define STB_IMAGE_IMPL
#include <stb_image.h>
#define STB_WRITE_IMAGE_IMPL
#include <stb_image_write.h>

VideoProcessing::VideoProcessing(const std::string& videoFilePath) : videoFilePath(videoFilePath) {
    // Initializing Q compression matrix:
    Q.resize(8, 8);
    Q << 16, 11, 10, 16, 24, 40, 51, 61,
         12, 12, 14, 19, 26, 58, 60, 55,
         14, 13, 16, 24, 40, 57, 69, 56,
         14, 17, 22, 29, 51, 87, 80, 62,
         18, 22, 37, 56, 68, 109, 103, 77,
         24, 35, 55, 64, 81, 104, 113, 92,
         49, 64, 78, 87, 103, 121, 120, 101,
         72, 92, 95, 98, 112, 100, 103, 99;

    // Initializing rows and cols of input frames:
    frame_rows = 0;
    frame_cols = 0;
}


void VideoProcessing::ExtractFrames(std::vector<Mat>& frames) {

    // Open the video utilising ffmpeg:
    AVFormatContext* formatContext = avformat_alloc_context();
    if (avformat_open_input(&formatContext, videoFilePath.c_str(), nullptr, nullptr) != 0) {
        return;
    }

    // Frame structure:
    AVFrame* frame = av_frame_alloc();
    int frameNumber = 0; 

    // Number of rows and cols is costant for each frame:
    frame_cols = frame->width;
    frame_rows = frame->height;

    // Extract frame from video until av_read_frame gives a negative value, it means that all frames have been read:
    while (av_read_frame(formatContext, frame) >= 0) {
        // utilizing stb_image.h takes frame in a integer matrix
        Mat matrix(frame_rows, frame_cols);

        // Add matrix to frames vector:
        frames.push_back(matrix);

        // Save frame in version .jpg in 'frames' directory:
        std::string frameFileName = "../input_frames/frame_" + std::to_string(frameNumber++) + ".jpg";
        stbi_write_jpg(frameFileName.c_str(), frame_cols , frame_rows, 3, frame->data[0], 100);

        // Free the frame:
        av_frame_unref(frame);
    }

    // Free resources:
    avformat_close_input(&formatContext);
    av_frame_free(&frame);
}


void VideoProcessing::DivideIntoBlocks(const std::vector<Mat>& frames, std::vector<Mat>& blocks) {
    for (const Mat& frame : frames) {
        int numRows = frame.rows();
        int numCols = frame.cols();

        // Frame dimensions must be multiple of 8: 
        int numBlockRows = numRows / 8;
        int numBlockCols = numCols / 8;

        // Subdivide the frame in blocks:
        for (int i = 0; i < numBlockRows; ++i) {
            for (int j = 0; j < numBlockCols; ++j) {
                //Eigen::Block<Derived, Rows, Cols> MatrixType::block(Index startRow, Index startCol, Index numRows, Index numCols);
                Mat block = frame.block(i * 8, j * 8, 8, 8);
                blocks.push_back(block);
            }
        }
    }
}


void VideoProcessing::Subtract128(std::vector<Mat>& blocks) {
    for (int i = 0; i < blocks.size(); i++) {
        Mat& block = blocks[i];
        int numRows = block.rows();
        int numCols = block.cols();

        for (int j = 0; j < numRows; j++) {
            for (int k = 0; k < numCols; k++) {
                block(j, k) -= 128;
            }
        }
    }
}


void ApplyDCT(const std::vector<Mat>& input_blocks, std::vector<Mat>& frequency_blocks) {
    // Vector of matrices with complex double entries must be empty
    frequency_blocks.clear();

    for (const Mat& input_block : input_blocks) {
        // Create new block with same dimensions
        Mat output_block = input_block;

        // DCT on input block:
        DCT_2D dct(input_block, output_block);
        dct.transform_par();

        // Add new block to vector of complex double entries matrices
        frequency_blocks.push_back(output_block);
    }
}



void VideoProcessing::Quantization(std::vector<Mat>& frequency_blocks) {
    // For each block in vector of frequency blocks, apply quantization + rounding to nearest integer
    for (Mat& block : frequency_blocks) {
        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                block(i, j) = std::ceil(block(i,j) / Q(i,j) );
            }
        }
    }

}

void VideoProcessing::DecodingBlocks(std::vector<Mat>& frequency_blocks){
    // Loop through blocks in vector of frequency blocks and reconstruct each quantized 8x8 block by multiply elementwise by Q:
    for (Mat& block : blocks) {
        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                block(i, j) = block(i, j) * Q(i, j);
            }
        }
    }
}

void VideoProcessing::InverseDCT(std::vector<Mat>& frequency_blocks) {
    // Loop through each block and apply the inverse DCT
    for (Mat& block : frequency_blocks) {
        // Create a new block for the inverse transform:
        Mat inverse_block(block.rows(), block.cols());

        DCT_2D dct(block, inverse_block);
        dct.iTransform();

        // Replace the original block with the inverse transform:
        block = inverse_block;

        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                // Round to the nearest integer:
                int rounded_value = std::round(block(i, j));

                // Add 128 and allocate the value to the element of the block:
                rounded_value += 128;
                // Set the real part of the element to the rounded value:
                block(i, j) = rounded_value;
            }
        }          
    }
}

void VideoProcessing::ReconstructFrame(std::vector<Mat>& frequency_blocks) {
    // Ricostruire il frame passando per i blocchi di frequency blocks
}

void VideoProcessing::SaveVideo(const std::string& outputVideoFilePath, const std::vector<Mat>& frames) {
    // salvataggio dei frame come video utilizzando stb_image_write.h
}
