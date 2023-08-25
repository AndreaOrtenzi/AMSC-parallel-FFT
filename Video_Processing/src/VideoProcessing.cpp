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
    numFrames  = 0;
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
    numFrames = 0;

    // Extract frame from video until av_read_frame gives a negative value, it means that all frames have been read:
    while (av_read_frame(formatContext, frame) >= 0) {
        // Create a matrix and copy frame data into it:
        Mat matrix(frame->height, frame->width);
        for (int x = 0; x < frame->height; ++x) {
            for (int y = 0; y < frame->width; ++y) {
                matrix(x, y) = frame->data[0][x * frame->linesize[0] + y];
            }
        }

        // Add matrix to frames vector:
        frames.push_back(matrix);

        // Save frame as a JPEG image in the 'frames' directory:
        std::string frameFileName = "../input_frames/frame_" + std::to_string(frameNumber++) + ".jpg";
        stbi_write_jpg(frameFileName.c_str(), frame->width, frame->height, 1, matrix.data(), 100);

        // Increase the number of frames read:
        numFrames++;

        // Free the frame:
        av_frame_unref(frame);
    }

    // Number of frames read:
    std::cout << numFrames << " frames have been read. " << << std::endl;


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
                block(j, k).real() -= 128;
                // Set to zero imaginary part:
                block(j, k).imag() = 0.0;
            }
        }
    }
}


void ApplyFFT(const std::vector<Mat>& input_blocks, std::vector<Mat>& frequency_blocks) {
    // Vector of matrices with complex double entries must be empty
    frequency_blocks.clear();

    for (const Mat& input_block : input_blocks) {
        // Create new block with same dimensions
        Mat output_block = input_block;

        // FFT on input block:
        FFT_2D fft(input_block, output_block);
        fft.transform_par();

        // Add new block to vector of frequencies matrix
        frequency_blocks.push_back(output_block);
    }
}



void VideoProcessing::Quantization(std::vector<Mat>& frequency_blocks) {
    // For each block in vector of frequency blocks
    for (Mat& block : frequency_blocks) {
        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                // Apply quantization + rounding to nearest integer, ignoring the imaginary part:
                block(i, j) = static_cast<int>( std::round(block(i, j) / Q(i, j)) );
            }
        }
    }

}

void VideoProcessing::DecodingBlocks(std::vector<Mat>& frequency_blocks){
    // Loop through blocks in vector of frequency blocks and reconstruct each quantized 8x8 block by multiply elementwise by Q:
    for (Mat& block : blocks) {
        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                block(i, j) = static_cast<int>(block(i, j) * Q(i, j));
            }
        }
    }
}

void VideoProcessing::InverseFFT(std::vector<Mat>& frequency_blocks) {
    // Loop through each block and apply the inverse FFT for each block in vector:
    for (Mat& block : frequency_blocks) {
        // Create a new block for the inverse transform:
        Mat inverse_block(block.rows(), block.cols());

        FFT_2D fft(block, inverse_block);
        fft.iTransform();

        // Replace the original block with the inverse transform:
        block = inverse_block;

        for (int i = 0; i < block.rows(); i++) {
            for (int j = 0; j < block.cols(); j++) {
                // Round to the nearest integer:
                int rounded_value = std::round(block(i, j));

                // Add 128 and allocate the value to the element of the block:
                rounded_value += 128;
                // Set the real part of the element to the rounded value:
                block(i, j).real() = rounded_value;
                block(i, j).imag() = 0;
            }
        }          
    }
}

void VideoProcessing::ReconstructFrames(std::vector<Mat>& frequency_blocks) {
    // Total number of blocks must be consistent with the number of frames:
    int num_blocks_per_frame = (frame_rows / 8) * (frame_cols / 8);
    int expected_total_blocks = num_blocks_per_frame * numFrames;

    if (frequency_blocks.size() != expected_total_blocks) {
        std::cerr << "Dimension of frequency blocks'vector is not consistent with the expected number of total blocks!" << std::endl;
        return;
    }

    // Create matrices'vector for reconstructed frames with size equal to number of frames
    std::vector<Mat> reconstructed_frames(numFrames);

    // Index to track blocks in the frequency_blocks vector
    int block_index = 0;

    for (int frame_index = 0; frame_index < numFrames; ++frame_index) {
        // Create matrix for current frame:
        Mat& current_frame = reconstructed_frames[frame_index];
        current_frame.resize(frame_rows, frame_cols);

        // Assembly blocks in current frame:
        for (int i = 0; i < frame_rows; i += 8) {
            for (int j = 0; j < frame_cols; j += 8) {
                // Copy the current block into the current frame:
                for (int x = 0; x < 8; x++) {
                    for (int y = 0; y < 8; y++) {
                        current_frame(i + x, j + y) = frequency_blocks[block_index](x, y);
                    }
                }
                block_index++;
            }
        }
    }

    // Check for dimension of reconstructed_frames vector: 
    if (reconstructed_frames.size() != numFrames){
        std::cerr << "Reconstructed frames' vector has dimension  different by the expected one. " <<std::endl;
        return;
    }

}


void SaveVideo(const std::string& outputVideoFilePath, const std::vector<Mat>& reconstructed_frames) {
    // Output folder for JPEG images
    const std::string outputImagesFolder = "../output_frames/";

    // Save JPEG images:
    for (int frameIndex = 0; frameIndex < reconstructed_frames.size(); frameIndex++) {
        const Mat& frame = reconstructed_frames[frameIndex];

        // Filename for JPEG image
        std::string frameFileName = outputImagesFolder + "frame_" + std::to_string(frameIndex) + ".jpg";

        // Create a matrix of 8-bit integers to save the JPEG image
        Eigen::MatrixXi jpegFrame(frame.rows(), frame.cols());
        for (int i = 0; i < frame.rows(); i++) {
            for (int j = 0; j < frame.cols(); j++) {
                // Truncate the imaginary part and round the real part:
                jpegFrame(i, j) = std::round(frame(i, j).real());
            }
        }

        // Convert integer matrix and save it as image with stbi_write:
        stbi_write_jpg(frameFileName.c_str(), jpegFrame.cols(), jpegFrame.rows(), 1, jpegFrame.data(), 100);
    }

    // Reconstruct the .mp4 video using ffmpeg.h:
    const std::string ffmpegCmd = "ffmpeg -framerate 30 -i " + outputImagesFolder + "frame_%d.jpg -c:v libx264 -pix_fmt yuv420p " + outputVideoFilePath;
    int ffmpegResult = system(ffmpegCmd.c_str());
    if (ffmpegResult != 0) {
        std::cerr << "Error during the creation of the video with ffmpeg: " << strerror(errno) << endl;
    }else{
        std::cout << "Compressed video successfully created:" << outputVideoFilePath << endl;
        }
}




    