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
  int padded_length = ((length + (512*32 - 1))/(1.0*512*32)) * (512*32);
  cudaMalloc((void**)&gpuA, padded_length);
  cudaMemset(&gpuA, 0, padded_length);
  cudaMalloc((void**)&gpuB, padded_length);
  cudaMemset(&gpuB, 0, padded_length);
  cudaMalloc((void**)&gpuC, padded_length);
  cudaMemset(&gpuC, 0, padded_length);

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
  cudaMemcpy(gpuA, a, sizeof(float) * padded_length, cudaMemcpyHostToDevice);
  cudaMemcpy(gpuB, b, sizeof(float ) * padded_length, cudaMemcpyHostToDevice);
  dim3 numThreads(512, 1);
  dim3 numBlocks(32, 1);
  cudaEventRecord(start, NULL);
  pair_wise_product<<<numBlocks, numThreads>>>(gpuA, gpuB, gpuC);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  //cudaEventRecord(start, NULL);
  cudaMemcpy(copyC, gpuC, Size, cudaMemcpyDeviceToHost);
  /*for (i=0; i < length; i++) {
    printf("%f ", copyC[i]);
  }
  printf("\n");*/
  cudaEventRecord(start, NULL);
  reduction_num_3<<<numBlocks, numThreads>>>(gpuC);
  cudaDeviceSynchronize();
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);

  cudaMemcpy(copyC, gpuC, Size, cudaMemcpyDeviceToHost);
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
