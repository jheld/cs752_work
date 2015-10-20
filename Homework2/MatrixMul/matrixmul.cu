#include <stdio.h>
#include <stdlib.h>
#include "matrixmul.h"

// include the kernel
#include "matrixmul_kernel.cu"

extern void computeGold(float*, const float*, const float*, unsigned int,
			unsigned int, unsigned int);

void gpuInit(void){
  cudaSetDevice(0);
}

void randomInit(float* data, int size){
  for (int i = 0; i < size; ++i)
    data[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char** argv){
  int WA = 4; // Matrix A width
  int HA = 6; // Matrix A height
  int WB = 4; // Matrix B width
  int HB; // Matrix B height
  int WC; // Matrix C width 
  int HC; // Matrix C height
  int i;

  i = 1;
  while (i<argc){
    if (!strcmp(argv[i],"wa")) sscanf(argv[i+1],"%d", &WA);
    else if (!strcmp(argv[i],"ha")) sscanf(argv[i+1],"%d", &HA);
    else if (!strcmp(argv[i],"wb")) sscanf(argv[i+1],"%d", &WB);
    i += 2;
  }
  //WA = WA * BLOCK_SIZE;
  //HA = HA * BLOCK_SIZE;
  //WB = WB * BLOCK_SIZE;
  HB = WA;
  WC = WB;
  HC = HA;

  gpuInit();
  srand(2006);
  printf("A: %d by %d\nB: %d by %d\n", HA, WA, HB, WB);

  unsigned int size_A = WA * HA;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)malloc(mem_size_A);
  unsigned int size_B = WB * HB;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)malloc(mem_size_B);
  unsigned int size_C = WC * HC;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float* h_C    = (float*) malloc(mem_size_C);
  float* h_Copy = (float*) malloc(mem_size_C);

  randomInit(h_A, size_A);
  randomInit(h_B, size_B);
  
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float msecTotal;

  cudaEventRecord(start, NULL);
  computeGold(h_Copy, h_A, h_B, HA, WA, WB);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("cpu time: %.3f ms\n", msecTotal);

  float* d_A, *d_B, *d_C;
  size_t pitch_a, pitch_b, pitch_c;
  cudaMallocPitch((void**) &d_A, &pitch_a, WA, HA);
  cudaMallocPitch((void**) &d_B, &pitch_b, WB, HB);
  cudaMallocPitch((void**) &d_C, &pitch_c, WC, HC);

  int normalized_wa = (((WA + BLOCK_SIZE - 1)/(BLOCK_SIZE*1.0)) * BLOCK_SIZE);
  int normalized_ha = (((HA + BLOCK_SIZE - 1)/(BLOCK_SIZE*1.0)) * BLOCK_SIZE);
  int normalized_wb = (((WB + BLOCK_SIZE - 1)/(BLOCK_SIZE*1.0)) * BLOCK_SIZE);
  int normalized_hb = (((HB + BLOCK_SIZE - 1)/(BLOCK_SIZE*1.0)) * BLOCK_SIZE);
  int normalized_wc = (((WC + BLOCK_SIZE - 1)/(BLOCK_SIZE*1.0)) * BLOCK_SIZE);
  int normalized_hc = (((HC + BLOCK_SIZE - 1)/(BLOCK_SIZE*1.0)) * BLOCK_SIZE);


  cudaEventRecord(start, NULL);

  cudaMemcpy2D(d_A, normalized_wa*sizeof(int), h_A, pitch_a, normalized_wa*sizeof(int), normalized_ha, cudaMemcpyHostToDevice);
  cudaMemset2D(&d_A, normalized_wa*sizeof(int), 0, normalized_wa, normalized_ha);
  cudaMemcpy2D(d_B, normalized_wb*sizeof(int), h_B, pitch_b, normalized_wb*sizeof(int), normalized_hb, cudaMemcpyHostToDevice);
  cudaMemset2D(&d_B, WB*sizeof(int), 0, normalized_wb, normalized_hb);
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(WC/threads.x, HC/threads.y);
  matrixMul<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);
  cudaDeviceSynchronize();
  cudaMemcpy2D(d_C, normalized_wc*sizeof(int), h_C, pitch_c, normalized_wc*sizeof(int), normalized_hc, cudaMemcpyDeviceToHost);
  cudaMemset2D(&d_C, normalized_wc*sizeof(int), 0, normalized_wc, normalized_hc);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("gpu time: %.3f ms\n", msecTotal);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Copy);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return 0;
}
