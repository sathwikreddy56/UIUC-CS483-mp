#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 8
//@@ Define constant memory for device kernel here
__constant__ float M[MASK_WIDTH * MASK_WIDTH * MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  int ix = bx * blockDim.x + tx;
  int iy = by * blockDim.y + ty;
  int iz = bz * blockDim.z + tz;
  
  /*int ndsWidth = TILE_WIDTH + MASK_WIDTH  - 1;
  __shared__ float N_ds[(TILE_WIDTH + MASK_WIDTH  - 1) 
                        * (TILE_WIDTH + MASK_WIDTH  - 1) 
                        * (TILE_WIDTH + MASK_WIDTH  - 1)];
  int n = MASK_WIDTH / 2;
  
  int halo_idx_left = (bx - 1) * blockDim.x + tx;
  if (tx > blockDim.x - n) {
    N_ds[tz * ndsWidth * ndsWidth + ty * ndsWidth + (tx - (blockDim.x - n))]
      = (halo_idx_left < 0) ? 0 : input[iz * x_size * y_size + iy * x_size + halo_idx_left];
  }
  
  int halo_idx_right = (bx + 1) * blockDim.x + tx;
  if (tx < n) {
    N_ds[tz * ndsWidth * ndsWidth + ty * ndsWidth + (tx + blockDim.x + n)]
      = (halo_idx_right >= x_size) ? 0 : input[iz * x_size * y_size + iy * x_size + halo_idx_right];
  }
  
  int halo_idx_front = (by - 1) * blockDim.y + ty;
  if (ty > blockDim.y - n) {
    N_ds[tz * ndsWidth * ndsWidth + (ty - (blockDim.y - n)) * ndsWidth + tx]
      = (halo_idx_front < 0) ? 0 : input[iz * x_size * y_size + halo_idx_front * x_size + ix];
  }
  
  int halo_idx_rear = (by + 1) * blockDim.y + ty;
  if (ty < n) {
    N_ds[tz * ndsWidth * ndsWidth + (ty + blockDim.y + n) * ndsWidth + tx]
      = (halo_idx_rear >= y_size) ? 0 : input[iz * x_size * y_size + halo_idx_front * x_size + ix];
  }
  
  int halo_idx_up = (bz - 1) * blockDim.z + tz;
  if (tz > blockDim.z - n) {
    N_ds[(tz - (blockDim.z - n)) * ndsWidth * ndsWidth + ty * ndsWidth + tx]
      = (halo_idx_up < 0) ? 0 : input[halo_idx_up * x_size * y_size + iy * x_size + ix];
  }
  
  int halo_idx_down  = (by + 1) * blockDim.y + ty;
  if (ty < n) {
    N_ds[(tz + blockDim.z + n) * ndsWidth * ndsWidth + ty * ndsWidth + tx]
      = (halo_idx_down >= z_size) ? 0 : input[halo_idx_up * x_size * y_size + iy * x_size + ix];
  }
  N_ds[(n + tz) * ndsWidth * ndsWidth + (n + ty) * ndsWidth + (n + tx)]
    = input[iz * x_size * y_size + iy * x_size + ix];
  __syncthreads();*/
  
  printf("Block (%d, %d, %d), Thread (%d, %d, %d)\n", bx, by, bz, tx, ty, tz);
  
  float pValue = 0;
  int x_sp = ix - (MASK_WIDTH / 2);
  int y_sp = iy - (MASK_WIDTH / 2);
  int z_sp = iz - (MASK_WIDTH / 2);
  for (int i = 0; i < MASK_WIDTH; i++)
    for (int j = 0; j < MASK_WIDTH; j++)
      for (int k = 0; k < MASK_WIDTH; k++) {
        float t = 0;
        if (0 <= (z_sp + i) && (z_sp + i) < z_size &&
           0 <= (y_sp + j) && (y_sp + j) < y_size &&
           0 <= (x_sp + k) && (x_sp + k) < x_size) {
          t = input[(z_sp + i)*x_size*y_size + (y_sp + j)*x_size + (x_sp + k)];
        }
        pValue += t * M[i*MASK_WIDTH*MASK_WIDTH + j*MASK_WIDTH + k];
      }
  if (0 <= iz && iz < z_size &&
     0 <= iy && iy < y_size &&
     0 <= ix && ix < x_size) {
    printf("output[%d][%d][%d] = %.4f\n", iz, iy, ix, pValue);
    output[iz*x_size*y_size + iy*x_size + ix] = pValue;
  }
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, z_size * y_size * x_size * sizeof(float));
  cudaMalloc((void**) &deviceOutput, z_size * y_size * x_size * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do not need to be copied to the gpu
  cudaMemcpyToSymbol(M, hostKernel, MASK_WIDTH * MASK_WIDTH * MASK_WIDTH * sizeof(float));
  cudaMemcpy(deviceInput, &hostInput[3], z_size * y_size * x_size * sizeof(float), cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 gridDim(ceil(x_size / (TILE_WIDTH * 1.0)), 
               ceil(y_size / (TILE_WIDTH * 1.0)), 
               ceil(z_size / (TILE_WIDTH * 1.0)));
  dim3 blockDim(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
  //@@ Launch the GPU kernel here
  conv3d<<<gridDim, blockDim>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(&hostOutput[3], deviceOutput, z_size * y_size * x_size * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}


