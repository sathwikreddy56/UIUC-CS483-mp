# Parallel Programming Coursework: Machine Problems

This is the course labs for ECE408/CS483 *Parallel Programming* at University of Illinois at Urbana-Champaign in Fall 2018. All students originally develop their assignments using [WebGPU](https://www.webgpu.net).

There are 8 machine problems (MP) for the whole semester, each one with an objective. Dataset are provided to develop the machine problem on your own system or to examine the data. These dataset can be downloaded [here](https://drive.google.com/open?id=1wshhMhg5Kcg6BvIG14r1AQjgaipIi4vX).

## Machine Problems

### MP0. Lab Tour with Device Query
The purpose of this lab is to get you familiar with using the submission system for this course and the hardware used.

### MP1. Vector Addition
The purpose of this lab is to get you familiar with using the CUDA API by implementing a simple vector addition kernel and its associated host code as shown in the lectures.

### MP2. Basic Matrix Multiplication
The purpose of this lab is to implement a basic dense matrix multiplication routine.

### MP3. Tiled Matrix Multiplication
The purpose of this lab is to implement a tiled dense matrix multiplication routine using shared memory.

### MP4. 3D Convolution
The purpose of this lab is to implement a 3D convolution using constant memory for the kernel and 3D shared memory tiling.

### MP5.1. List Reduction
Implement a kernel and associated host code that performs reduction of a 1D list stored in a C array. The reduction should give the sum of the list. You should implement the improved kernel discussed in the lecture. Your kernel should be able to handle input lists of arbitrary length.

### MP5.2. Parallel Scan
The purpose of this lab is to implement one or more kernels and their associated host code to perform parallel scan on a 1D list. The scan operator used will be addition. You should implement the work- efficient kernel discussed in lecture. Your kernel should be able to handle input lists of arbitrary length. However, for simplicity, you can assume that the input list will be at most 2,048 * 2,048 elements.

### MP6. Histogram Equalization
The purpose of this lab is to implement an efficient histogramming equalization algorithm for an input image. Like the image convolution MP, the image is represented as `RGB float` values. You will convert that to `GrayScale unsigned char` values and compute the histogram. Based on the histogram, you will compute a histogram equalization function which you will then apply to the original image to get the color corrected image.

### MP7. Sparse Matrix Multiplication (JDS)
The purpose of this lab is to implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based on the Jagged Diagonal Storage (JDS) transposed format.

## About the course (Fall 2018)
- [Illinois WikiPage](https://wiki.illinois.edu/wiki/display/ECE408/ECE408F18+Home)
- Instructor: [Sanjay J. Patel](https://ece.illinois.edu/directory/profile/sjp)
- Location: 2079 Natural History Building, UIUC
