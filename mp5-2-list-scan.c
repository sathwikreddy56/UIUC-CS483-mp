// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 1024 //@@ You can change this
#define SECTION_SIZE 2*BLOCK_SIZE

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  __shared__ float XY[SECTION_SIZE];
  int i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    XY[threadIdx.x] = input[i];
    printf("block %d: XY[%d] = input[%d] = %.2f\n", blockIdx.x, 
         threadIdx.x, i, input[i]);
  }
  if (i + blockDim.x < len) {
    XY[threadIdx.x + blockDim.x] = input[i + blockDim.x];
    printf("block %d: XY[%d] = input[%d] = %.2f\n", blockIdx.x, 
         threadIdx.x + blockDim.x, i + blockDim.x, input[i + blockDim.x]);
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
    printf("block %d: Y[%d] = XY[%d] = %.2f\n", blockIdx.x, 
         i, threadIdx.x, XY[threadIdx.x]);
  }
  if (i + blockDim.x < len) {
    output[i + blockDim.x] = XY[threadIdx.x + blockDim.x];
    printf("block %d: Y[%d] = XY[%d] = %.2f\n", blockIdx.x, 
         i + blockDim.x, threadIdx.x + blockDim.x, XY[threadIdx.x + blockDim.x]);
  }
}

__global__ void add(float* output, int len) {
  __shared__ float sumPrevBlocks;
  if (threadIdx.x == 0) {
    sumPrevBlocks = 0;
    for (int i = 0; i < blockIdx.x / 2; i++) {
      if (i * SECTION_SIZE + SECTION_SIZE - 1 < len)
        sumPrevBlocks += output[i * SECTION_SIZE + SECTION_SIZE - 1];
    }
    printf("block %d: sumPrevBlocks = %.1f\n", blockIdx.x, sumPrevBlocks);
  }
  __syncthreads();
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    output[i] += sumPrevBlocks;
    printf("output[%d] = %.1f\n", i, output[i]);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(BLOCK_SIZE*1.0)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  printf("len = %d\n", numElements);
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numElements);
  add<<<dimGrid, dimBlock>>>(deviceOutput, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}

