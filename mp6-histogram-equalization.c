// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32
#define SECTION_SIZE 2*HISTOGRAM_LENGTH

//@@ insert code here
__global__ void grayScale(float* colored, int* gray, int* hist, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int idx = y*width + x;
    float r = colored[3*idx];
    float g = colored[3*idx + 1];
    float b = colored[3*idx + 2];
    gray[idx] = (0.21*r + 0.71*g + 0.07*b) * 255;
    atomicAdd(&(hist[gray[idx]]), 1);
  }
}

__global__ void scan(int *input, int *output, int len) {
  __shared__ int XY[SECTION_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    XY[threadIdx.x] = input[i];
    // printf("hist[%d]: %d\n", i, XY[threadIdx.x]);
  }
  if (i + blockDim.x < len) {
    XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
  }

  for (int stride = 1; stride <= SECTION_SIZE; stride *= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride - 1;
    if (index < SECTION_SIZE) {
      XY[index] += XY[index - stride];
    }
  }
  for (int stride = SECTION_SIZE/4; stride > 0; stride /= 2) {
    __syncthreads();
    int index = (threadIdx.x + 1) * 2 * stride -1;
    if (index + stride < SECTION_SIZE) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if (i < len) {
    output[i] = XY[threadIdx.x];
    // printf("cdf[%d]: %d\n", i, output[i]);
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
  }
}

__global__ void correctColor(float* input, float* output, int* cdf, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    for (int channel = 0; channel < 3; channel++) {
      int idx = 3 * (y*width + x) + channel;
      int color = int (input[idx] * 255.0);
      int corrected_color = 255 * (cdf[color] - cdf[0]) / (width * height - cdf[0]);
      if (corrected_color < 0) corrected_color = 0;
      if (corrected_color > 255) corrected_color = 255;
      output[idx] = corrected_color / 255.0;
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *d_inputImageData;
  int *d_grayImageData;
  float *d_outputImageData;
  int *d_histogram;
  int *d_cdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  cudaMalloc((void **)&d_inputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void **)&d_grayImageData, imageWidth*imageHeight*sizeof(int));
  cudaMalloc((void **)&d_outputImageData, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void **)&d_histogram, HISTOGRAM_LENGTH*sizeof(int));
  cudaMalloc((void **)&d_cdf, HISTOGRAM_LENGTH*sizeof(int));
  cudaMemcpy(d_inputImageData, hostInputImageData,
             imageWidth*imageHeight*imageChannels*sizeof(float),
             cudaMemcpyHostToDevice);

  dim3 gridDim(ceil(imageWidth / (BLOCK_SIZE * 1.0)), ceil(imageHeight / (BLOCK_SIZE * 1.0)), 1);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE, 1);
  grayScale<<<gridDim, blockDim>>>(d_inputImageData, d_grayImageData, d_histogram, imageWidth, imageHeight);

  dim3 gridDimCalcCdf(1, 1, 1);
  dim3 blockDimCalcCdf(HISTOGRAM_LENGTH, 1, 1);
  scan<<<gridDimCalcCdf, blockDimCalcCdf>>>(d_histogram, d_cdf, HISTOGRAM_LENGTH);

  dim3 gridDimCorrect(ceil(imageWidth / (BLOCK_SIZE * 1.0)), ceil(imageHeight / (BLOCK_SIZE * 1.0)), 1);
  dim3 blockDimCorrect(BLOCK_SIZE, BLOCK_SIZE, 1);
  correctColor<<<gridDim, blockDim>>>(d_inputImageData, d_outputImageData, d_cdf, imageWidth, imageHeight);

  cudaMemcpy(hostOutputImageData, d_outputImageData,
             imageWidth*imageHeight*imageChannels*sizeof(float),
             cudaMemcpyDeviceToHost);
  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(d_inputImageData);
  cudaFree(d_grayImageData);
  cudaFree(d_outputImageData);
  cudaFree(d_histogram);
  cudaFree(d_cdf);

  return 0;
}
