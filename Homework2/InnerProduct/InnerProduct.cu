#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

__global__ void pair_wise_product(float *a, float *b, float *c) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  c[i] = a[i] * b[i];
}

__global__ void vectorSum(float *a, float *b, float *c){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  c[i] = a[i] + b[i];
}


__global__ void reduction_num_3(float *c) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
    if (tid < s) {
      c[tid] += c[tid + s];
    }
    __syncthreads();
  }
}


int main(int argc, char *argv[]){
  unsigned int length = 4194304;
  int i, Size;
  float *a, *b, *c, *copyC, *gpuA, *gpuB, *gpuC;
  time_t seed;
  cudaEvent_t start;
  cudaEvent_t stop;
  float msecTotal;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  if (argc>1) {
    sscanf(argv[1],"%d",&length);
  }
  Size = sizeof(float)*length;
  a = (float *)calloc(length, sizeof(float));
  b = (float *)calloc(length, sizeof(float));
  c = (float *)calloc(length, sizeof(float));
  copyC = (float *)calloc(length, sizeof(float));
  time(&seed);
  srand48(seed);
  for (i=0; i<length; i++)
    a[i] = drand48(), b[i] = drand48();

  cudaSetDevice(0);
  int padded_length = floor((length + (512*32 - 1))/(1.0*512*32)) * (512*32);
  cudaError_t error;
  error = cudaMalloc((void**)&gpuA, padded_length*sizeof(float));
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMemset(gpuA, 0, padded_length*sizeof(float));
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void**)&gpuB, padded_length*sizeof(float));
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMemset(gpuB, 0, padded_length*sizeof(float));
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMalloc((void**)&gpuC, padded_length*sizeof(float));
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMemset(gpuC, 0, padded_length*sizeof(float));
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(start, NULL);
  for (i=0; i<length; i++)
    c[i] = a[i] * b[i];
  for (i = 1; i < length; i++) {
    c[0] += c[i];
  }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("cpu time: %.3f ms\n", msecTotal);
  error = cudaMemcpy(gpuA, a, Size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMemcpy(gpuB, b, Size, cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  dim3 numThreads(512, 1);
  dim3 numBlocks(32, 1);
  cudaEventRecord(start, NULL);
  pair_wise_product<<<numBlocks, numThreads>>>(gpuA, gpuB, gpuC);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  //cudaEventRecord(start, NULL);
  error = cudaMemcpy(copyC, gpuC, Size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  /*for (i=0; i < length; i++) {
    printf("%f ", copyC[i]);
  }
  printf("\n");*/
  cudaEventRecord(start, NULL);
  reduction_num_3<<<numBlocks, numThreads>>>(gpuC);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  error = cudaMemcpy(copyC, gpuC, Size, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("oops, %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("gpu time: %.3f ms\n", msecTotal);
  //printf("%f\n", copyC[0]);
  for (i=0; i<length; i++)
    if (fabs(c[i]-copyC[i]) > 0.000001){
      printf("%d\t%f\t%f\n", i, c[i], copyC[i]);
      return 1;
    }

  return 0;
}
