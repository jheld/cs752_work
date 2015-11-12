#include <stdio.h>
#include <stdlib.h>
#include "matrixmul.h"
#include <math.h>

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
  cudaError_t error;
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
  //unsigned int mem_size_A = sizeof(float) * size_A;
  float* h_A = (float*)calloc(size_A, sizeof(float));
  unsigned int size_B = WB * HB;
  //unsigned int mem_size_B = sizeof(float) * size_B;
  float* h_B = (float*)calloc(size_B, sizeof(float));
  unsigned int size_C = WC * HC;
  //unsigned int mem_size_C = sizeof(float) * size_C;
  float* h_C    = (float*) calloc(size_C, sizeof(float));
  float* h_Copy = (float*) calloc(size_C, sizeof(float));

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
  i = 0;
  int bound = 10;//HA * WA;
  /*printf("h_A:\n");
  for (; i < bound; i++) {
    printf("%f ", h_A[i]);
  }
  printf("\n");
  
  bound = 10;//HB * WB;
  printf("h_B:\n");
  i = 0;
  for (; i < bound; i++) {
    printf("%f ", h_B[i]);
  }
  printf("\n");*/

  printf("h_Copy:\n");
  i = 0;
  bound = 10;//WC * HC;
  for (; i < bound; i++) {
    printf("%f ", h_Copy[i]);
  }
  printf("\n");
  

  int reference_size = BLOCK_SIZE;
  int normalized_wa = (floor((WA + reference_size - 1)/(reference_size*1.0)) * reference_size);
  int normalized_ha = (floor((HA + reference_size - 1)/(reference_size*1.0)) * reference_size);
  int normalized_wb = (floor((WB + reference_size - 1)/(reference_size*1.0)) * reference_size);
  int normalized_hb = (floor((HB + reference_size - 1)/(reference_size*1.0)) * reference_size);
  int normalized_wc = (floor((WC + reference_size - 1)/(reference_size*1.0)) * reference_size);
  int normalized_hc = (floor((HC + reference_size - 1)/(reference_size*1.0)) * reference_size);
  printf("norm A: %d by %d\nnorm B: %d by %d\n", normalized_ha, normalized_wa, normalized_hb, normalized_wb);
  //printf("norm wa: %d\n", normalized_wa);
  float* d_A, *d_B, *d_C;
  size_t pitch_a, pitch_b, pitch_c;
  error = cudaMallocPitch((void**) &d_A, &pitch_a, normalized_wa*sizeof(float), normalized_ha); 
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  error = cudaMemset2D(d_A, pitch_a, 0, normalized_wa*sizeof(float), normalized_ha);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMallocPitch((void**) &d_B, &pitch_b, normalized_wb*sizeof(float), normalized_hb); 
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  error = cudaMemset2D(d_B, pitch_b, 0, normalized_wb*sizeof(float), normalized_hb);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMallocPitch((void**) &d_C, &pitch_c, normalized_wc*sizeof(float), normalized_hc); 
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  error = cudaMemset2D(d_C, pitch_c, 0, normalized_wc*sizeof(float), normalized_hc);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMemcpy2D(d_A, pitch_a, h_A, WA*sizeof(float), WA*sizeof(float), HA, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    {
      printf("cudaMemcpy2D d_C->h_C returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }

  error = cudaMemcpy2D(d_B, pitch_b, h_B, WB*sizeof(float), WB*sizeof(float), HB, cudaMemcpyHostToDevice);
  if (error != cudaSuccess)
    {
      printf("cudaMemcpy2D d_C->h_C returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }



  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid(normalized_wc/threads.x, normalized_hc/threads.y);
  matrixMul_pitch<<<grid, threads>>>(d_C, d_A, d_B, normalized_wa, normalized_wb, pitch_a, pitch_b);

  error = cudaDeviceSynchronize();
  if (error != cudaSuccess)
    {
      printf("cudaMemcpy2D d_C->h_C returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }

  error = cudaMemcpy2D(h_C, WC*sizeof(float), d_C, pitch_c, WC*sizeof(float), HC, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    {
      printf("cudaMemcpy2D d_C->h_C returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&msecTotal, start, stop);
  printf("cuda2d pitch strategy gpu time: %.3f ms\n", msecTotal);

  memset(h_A, 0, HA*WA*sizeof(float));
  error = cudaMemcpy2D(h_A, WA*sizeof(float), d_A, pitch_a, WA*sizeof(float), HA, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    {
      printf("cudaMemcpy2D d_C->h_C returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }
  /*
  printf("h_A copied back:\n");
  i = 0;
  bound = WA * HA;
  for (; i < bound; i++) {
    printf("%f ", h_A[i]);
  }
  printf("\n");
  
  */
  //cudaError_t error;
  memset(h_B, 0, HB*WB*sizeof(float));
  error = cudaMemcpy2D(h_B, WB*sizeof(float), d_B, pitch_b, WB*sizeof(float), HB, cudaMemcpyDeviceToHost);
  if (error != cudaSuccess)
    {
      printf("cudaMemcpy2D d_C->h_C returned error code %d, line(%d)\n", error, __LINE__);
      exit(EXIT_FAILURE);
    }
  /*
  printf("h_B copied back:\n");
  i = 0;
  bound = WB * HB;
  for (; i < bound; i++) {
    printf("%f ", h_B[i]);
  }
  printf("\n");
  */


  printf("h_C first 10 indices:\n");
  i = 0;
  bound = 10;//WC * HC;
  for (; i < bound; i++) {
    printf("%f ", h_C[i]);
  }
  printf("\n");
  


  error = cudaFree(d_A);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_B);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_C);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  unsigned int mem_size_A = sizeof(float) * normalized_wa * normalized_ha;
  error = cudaMalloc((void**) &d_A, mem_size_A);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  error = cudaMemset(d_A, 0, mem_size_A);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  unsigned int mem_size_B = sizeof(float) * normalized_wb * normalized_hb;
  error = cudaMalloc((void**) &d_B, mem_size_B);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  error = cudaMemset(d_B, 0, mem_size_B);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  unsigned int mem_size_C = sizeof(float) *  normalized_wc * normalized_hc;
  error = cudaMalloc((void**) &d_C, mem_size_C);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  error = cudaMemset(d_C, 0, mem_size_C);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(start, NULL);
  error = cudaMemcpy(d_A, h_A, ((unsigned int )WA*HA)*sizeof(float), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaMemcpy(d_B, h_B, ((unsigned int)WB*HB)*sizeof(float), cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }
  //  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  // dim3 grid(normalized_wc/threads.x, normalized_hc/threads.y);

  matrixMul<<<grid, threads>>>(d_C, d_A, d_B, WA, WB);
  error = cudaDeviceSynchronize();
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  memset(h_C, 0, WC * HC *sizeof(float));
  error = cudaMemcpy(h_C, d_C, ((unsigned int)WC * HC) * sizeof(float), cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaEventRecord(stop, NULL);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaEventSynchronize(stop);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaEventElapsedTime(&msecTotal, start, stop);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  printf("cpu strategy gpu time: %.3f ms\n", msecTotal);
  
  printf("h_C first 10 indices:\n");
  i = 0;
  bound = 10;//WC * HC;
  for (; i < bound; i++) {
    printf("%f ", h_C[i]);
  }
  printf("\n");



  error = cudaFree(d_A);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_B);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  error = cudaFree(d_C);
  if (error != cudaSuccess) {
    printf("oops, line: %d, error: %d\n", __LINE__, error);
    exit(EXIT_FAILURE);
  }

  //printf("norm wa: %d, norm wb: %d\n", normalized_wa, normalized_wb);

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_Copy);

  cudaDeviceReset();
  return 0;
}
