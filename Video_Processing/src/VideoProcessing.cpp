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


void VideoProcessing::ExtractFrames(std::vector<int_Mat>& frames) {

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
        int_Mat matrix(frame_rows, frame_cols);

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
    // Vector of matrices with complex double entries must be empty
    cd_blocks.clear(); 

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
    // Vector of matrices with complex double entries must be empty
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



void VideoProcessing::Quantization(std::vector<cd_Mat>& frequency_blocks) {
    // For each block in vector of frequency blocks, apply quantization + rounding to nearest integer
    for (cd_Mat& block : frequency_blocks) {
        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                block(i, j) = std::ceil(block(i,j) / Q(i,j) );
            }
        }
    }

}

void VideoProcessing::DecodingBlocks(std::vector<cd_Mat>& frequency_blocks){
    // Loop through blocks in vector of frequency blocks and reconstruct each quantized 8x8 block by multiply elementwise by Q:
    for (cd_Mat& block : blocks) {
        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                block(i, j) = block(i, j) * Q(i, j);
            }
        }
    }
}

void VideoProcessing::InverseFFT(std::vector<cd_Mat>& frequency_blocks) {
    // Loop through each block and apply the inverse FFT
    for (cd_Mat& block : frequency_blocks) {
        // Create a new block for the inverse transform:
        cd_Mat inverse_block(block.rows(), block.cols());

        FFT_2D fft(block, inverse_block);
        fft.inv_transform_par();

        // Replace the original block with the inverse transform:
        block = inverse_block;

        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                // Round the real part to the nearest integer:
                double rounded_value = std::round(block(i, j).real());

                // Add 128 and allocate the value to the real part of the element of the block:
                rounded_value += 128.0;
                // Set the real part of the element to the rounded value:
                block(i, j).real(rounded_value);
            }
        }          
    }
}

void VideoProcessing::NewConvertBlocks(std::vector<cd_Mat>& cd_blocks, std::vector<int_Mat>& new_frame_blocks) {
    // New vector of integer block matrices must be empty:
    new_frame_blocks.clear();

    for (const cd_Mat& cd_block : cd_blocks) {
        int_Mat int_block(cd_block.rows(), cd_block.cols());

        // Copy values converting from double to int:
        for (int i = 0; i < cd_block.rows(); ++i) {
            for (int j = 0; j < cd_block.cols(); ++j) {
                // Ignore imaginary part:
                int_block(i, j) = static_cast<int>(cd_block(i, j).real());
            }
        }

        // Push new block to vector:
        new_frame_blocks.push_back(int_block);
    }
}


void VideoProcessing::ReconstructFrame(std::vector<int_Mat>& new_frame_blocks) {

}

void VideoProcessing::SaveVideo(const std::string& outputVideoFilePath, const std::vector<int_Mat>& frames) {
    // salvataggio dei frame come video utilizzando stb_image_write.h
}
