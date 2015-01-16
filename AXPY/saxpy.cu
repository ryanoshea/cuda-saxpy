#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <Windows.h>

__global__ void saxpy(float *result, float *a, float *X, float *Y)
{
  // Linearize the thread arrangement
  int idx = blockIdx.x * (blockDim.x * blockDim.y * blockDim.z) 
          + threadIdx.z * (blockDim.y * blockDim.x) 
          + threadIdx.y * blockDim.x 
          + threadIdx.x;
  result[idx] = *a * X[idx] + Y[idx];
}

// Performs the single-precision AXPY operation (aX + Y), 
// where a is scalar and X,Y are vectors.
int main(int argc, char** argv)
{
  const int VECTOR_SIZE = 65536;
  const int VECTOR_BYTES = VECTOR_SIZE * sizeof(float);

  // Define a, X, and Y
  float h_a = 432.847;
  float *h_X = (float *)malloc(VECTOR_BYTES);
  float *h_Y = (float *)malloc(VECTOR_BYTES);
  float *h_result = (float *)malloc(VECTOR_BYTES); // result
  for (int i = 0; i < VECTOR_SIZE; i++)
  {
    h_X[i] = i;
    h_Y[i] = VECTOR_SIZE - i;
  }

  // Begin device operations

  // Allocate space in device memory for a, X, Y, and result
  float *d_a, *d_X, *d_Y, *d_result;
  cudaMalloc((void **)&d_a, sizeof(float));
  cudaMalloc((void **)&d_X, VECTOR_BYTES);
  cudaMalloc((void **)&d_Y, VECTOR_BYTES);
  cudaMalloc((void **)&d_result, VECTOR_BYTES);

  // Transfer a, X, Y to device
  cudaMemcpy(d_a, &h_a, sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_X, h_X, VECTOR_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y, h_Y, VECTOR_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_result, h_result, VECTOR_BYTES, cudaMemcpyHostToDevice);

  // Start timer
  LARGE_INTEGER *startTime, *endTime, *freq;
  freq = (LARGE_INTEGER *)malloc(sizeof(LARGE_INTEGER));
  startTime = (LARGE_INTEGER *)malloc(sizeof(LARGE_INTEGER));
  endTime = (LARGE_INTEGER *)malloc(sizeof(LARGE_INTEGER));
  QueryPerformanceFrequency(freq);
  QueryPerformanceCounter(startTime);

  // Spawn threads
  saxpy<<<128, dim3(16,16,2)>>>(d_result, d_a, d_X, d_Y);

  // End timer
  QueryPerformanceCounter(endTime);
  __int64 elapsed = (*endTime).QuadPart - (*startTime).QuadPart;
  float elapsedTime = float(elapsed) / float((*freq).QuadPart);

  // Transfer result back to host
  cudaMemcpy(h_result, d_result, VECTOR_BYTES, cudaMemcpyDeviceToHost);
  
  // Display snippet of result
  printf("[%f,\n", h_result[0]);
  printf(" %f,\n", h_result[1]);
  printf(" %f,\n", h_result[2]);
  printf(" %f,\n", h_result[3]);
  printf(" %f,\n", h_result[4]);
  printf(" ...\n");
  printf(" %f]\n", h_result[VECTOR_SIZE - 1]);
  printf("\n");
  printf("Elapsed time (s): %f\n", elapsedTime);

  // Free objects
  cudaFree(d_a);
  cudaFree(d_result);
  cudaFree(d_X);
  cudaFree(d_Y);
  free(h_X);
  free(h_Y);
  free(h_result);

  exit(EXIT_SUCCESS);
}