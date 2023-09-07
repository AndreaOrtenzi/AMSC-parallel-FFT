# Adavanced Method for Scientific Computing - Fast Fourier Transform

## Authors
* Andrea Ortenzi
* Emanuele Broggini

## Overview  
This repository hosts the implementation for the algortihm of the Fast Fourier Transform. It's designed to provide to user
a complete view of the algorithm  ranging from the one-dimensional version to the two-dimensional version, with an application of FFT2D on images. In the repository there are six folders: one is related to the unidimensional transform (**FFT1D**); two are related to the bidimensional version (**FFT2D** and **FFT2D_Vec**) and in *Implementations > FFT2D* will be clarified the differences among the two; one is related to the inclusion of the useful libaries utilized in the project (**lib**); one is created for plotting computational times for FFT2D implementations and making comparisons between the two classes in the two-dimensional cases; and one is dedicated to the application of the Fast Fourier Transform for image compression (**IMG_compression**).

## Requirements

- **Programming Language**: C++ 
- **Dependencies**:
  - Eigen 3.3+

Additional libraries required for this project are included in the **lib** folder and do not need to be downloaded separately.


## FFT 1D
The folder includes an **inc** folder containing all the classes used to implementing the unidimensional FFT, a **src** folder that includes all the .cpp files that contain the code for classes methods and variables and a **main_test.cpp** file for testing. In **FFT1D** folder we have several classes to deal with sequential and parallel FFT. For parallel processing, two separate classes have been implemented to leverage OpenMP and MPI parallel facilities. Every file in the main directory has been extensively commented to facilitate an easy understanding of both the methods and the variables used. 

## Compilation
In order to compile we have provided a makefile in the folder.
First, run:

```    
make     
```

then, will be created two executables: **test_library.exe** has been created in order to demonstrate the usage of the FFT1D as a library, **test.exe** . At this point, you can run:

``` 
./test_library.exe  
```

and then:

```    
./test.exe -N [number that specifies the vector length] -iTT [number that specifies how many iterations you want perform for each class]     ```

Obviously, vector length provided must be a number power of 2 to perform FFT.

# FFT 2D

For the bidimensional-case we have two separated folders: **FFT2D** and **FFT2D_Vec**, the first one utilizes Eigen library, the other one not. Both of them utilise OpenMP for the parallel processing. The differences between the two will be highlighted below.

## FFT 2D with Eigen
The folder **FFT2D** contains the files useful to perform the Fast Fourier Transform on matrices making use of the Eigen library. An important premise is that this class has been designed to perform the transformations only on square matrices, not on rectangular ones.
This folder contains an **inc** folder containing the class used to implementing the bidimensional FFT and **parameters**, a file that contains declarations to define constants or conditional macros used within the source code. In **src** folder you could find a file with the implementations of class methods and a **main_test.cpp** file for testing. Every file in the directory has been extensively commented to facilitate an easy understanding of both the methods and the variables used. 

### Compilation
In order to compile we have provided a makefile in the folder.
First, run:

``` 
make   
```

then, 

```   
./test2D.exe -N [number that specifies the rows and columns length] -iTT [number that specifies how many iterations you want perform for each class] -nTH [number to set the number of threads used with OpenMP]   
```

## FFT 2D with vectors of vectors

Explain differences with FFT 2D Eigen.
Talk about Plot folder and what does it do.

# IMAGE COMPRESSION

As last goal of our project we have implemented an image compression using our FFT2D methods. 
## Image class
In the folder **inc** you can find the header class **Image**, is a crucial component of this project designed for image compression and decompression using the Fast Fourier Transform. It facilitates the entire image processing pipeline by managing the following tasks implemented in the file .cpp in the **src** folder:

- **Initialization**: When an Image object is created, it is initialized with parameters such as the paths to folders containing JPEG images (original and restored), the encoded folder path, the image name, and an optional boolean flag that determines whether to use compressed or original images for processing.

- **Reading Original Image**: If the isInputCompressed flag is set to false, the Image class reads the original JPEG image specified in the constructor. It extracts pixel data from the image, calculates the number of Minimum Coded Units (MCUs) required to cover the image, and creates these MCUs. Each MCU corresponds to a block of pixels in the image (8 x 8 pixels).

- **FFT Transformation**: The Image class provides methods for applying the FFT to the image. The transform method applies the FFT2D transformation to each MCU in parallel (utilizes std::execution::par from C++20 to parallelize the code), effectively converting pixel values into frequency domain values.

- **Inverse FFT Transformation**: To decompress the image, the Image class offers an iTransform method. This method performs the inverse FFT2D transformation on each MCU in parallel (utilizes std::execution::par from C++20 to parallelize the code), reverting frequency domain values back to pixel values.

- **Reading Compressed Image**: If the isInputCompressed flag is set to true, the Image class reads compressed data from the specified encoded folder. It retrieves metadata, including image width and height, and creates MCUs from the compressed data. This is part of the decompression process.

- **Writing Compressed Data**: The Image class also supports writing frequency values of the image to compressed files. It creates an output folder if it doesn't already exist, stores metadata, and iterates through each MCU to save their frequency values. This is typically performed during the compression process.

- **Writing Restored Image**: To produce the final restored image, the Image class assembles pixel data from the MCUs and saves it as an output JPEG image. This restored image is the result of the decompression process.

## MinimumCodedUnit class

In **inc** is present the **MinimumCodedUnit** class that is responsible for handling Minimum Coded Units (MCUs) in the image compression and decompression process. This class encapsulates the functionality needed to transform an MCU between pixel and frequency domains, handle compression and decompression, and read/write data to/from files. Here there are some explanations about the main methods and functionality of this class:

- **readImage**: This method reads image data from a buffer. It iterates through rows and columns of the MCU and channels (e.g., red, green, blue). Each channel value is extracted from the buffer and stored in the mcuValues matrix.

- **transform**: This method applies various transformations to the MCU. It first subtracts 128 from each pixel value in the MCU. Then, it applies the FFT 2D to the MCU. The result is then quantized by dividing by a quantization matrix Q. After this transformation, the MCU contains frequency domain values.

- **iTransform**: This method performs the inverse transformations to restore pixel values from frequency domain values. It first multiplies each element in the MCU by the corresponding element in the quantization matrix Q. Then, it applies the iFFT 2D to each channel to obtain the pixel values. The pixel values are rounded to the nearest integer, added by 128 and stored in the mcuValuesRestored matrix. The MCU is now in its original pixel domain form.

- **writeCompressedOnFile**: This method writes the compressed data (both phase and norm matrices) to files. It creates filenames based on the MCU index and channel. The phase and norm matrices are saved using the Eigen library in Matrix Market format.

- **readCompressedFromFile**: This method reads compressed data (both phase and norm matrices) from files. It reconstructs the phase and norm matrices using the Eigen library from Matrix Market files. The frequency domain values are now available in the normFreqDense and phaseFreqDense matrices.

- **FFT2DwithQuantization**: This method performs a FFT 2D with quantization on the MCU. It processes each channel separately.
After FFT, it quantizes the values using the quantization matrix Q.

- **writeImage**: This method writes the decompressed pixel data back to a buffer. It iterates through rows and columns of the MCU and channels. The pixel values from the mcuValuesRestored matrix are placed in the buffer. The final buffer contains pixel values for the image.

### Compilation
To test the compression, you have to upload in the folder **imgs** a JPEG image (we have provided two test images, one color image and one black and white image) and in the main file cpp specify correctly the image name and its path.
After that, you can run: 

```
make    
```

and then:

``` 
./main.exe  
```

It's important to set the exact number of channel to use in **parameters** file in **inc** folder: NUM_CHANNELS = 1 for black&white, NUM_CHANNELS = 3 for color images.


# Conclusions and results
-----
