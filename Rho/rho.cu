#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define BlockSize 16

void cpuPearson(float *input, int numRow, int numCol, float *output){
  int row, col, i;
  float x, y, sumX, sumY, sumX2, sumY2, sumXY;
  float avgX, avgY, varX, varY, cov, rho;

  for (row=0; row<numRow; row++){
    output[row*numRow + row] = 1.0;
    for (col=row+1; col<numRow; col++){
      sumX = sumY = sumX2 = sumY2 = sumXY = 0.0;
      for (i=0; i<numCol; i++){
	x = input[row*numCol + i];
	y = input[col*numCol + i];
	sumX += x;
	sumY += y;
	sumX2 += x*x;
	sumY2 += y*y;
	sumXY += x*y;
      }
      avgX = sumX / numCol;
      avgY = sumY / numCol;
      varX = (sumX2 - avgX*avgX*numCol) / (numCol-1);
      varY = (sumY2 - avgY*avgY*numCol) / (numCol-1);
      cov = (sumXY - avgX*avgY*numCol) / (numCol-1);
      rho = cov / sqrtf(varX*varY);
      output[row*numRow + col] = output[col*numRow + row] = rho;
    }
  }
}

__global__ void
gpuPearson(float *input, int numRow, int numCol, float *output){
  __shared__ float Xs[BlockSize][BlockSize];
  __shared__ float Ys[BlockSize][BlockSize]; 
  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int xBegin = bx * BlockSize * numCol;
  int yBegin = by * BlockSize * numCol;
  int yEnd = yBegin + numCol - 1;
  int x, y, k, outIdx;
  float sumX, sumY, sumX2, sumY2, sumXY;
  float avgX, avgY, varX, varY, cov, rho;
  
  sumX = sumY = sumX2 = sumY2 = sumXY = 0.0;
  for (y=yBegin, x=xBegin; y<=yEnd; y+=BlockSize, x+=BlockSize){
    Ys[ty][tx] = input[y + ty*numCol + tx];
    Xs[ty][tx] = input[x + ty*numCol + tx];
    __syncthreads();
    for (k=0; k<BlockSize; k++){
      sumX += Xs[tx][k];
      sumY += Ys[ty][k];
      sumX2 += Xs[tx][k] * Xs[tx][k];
      sumY2 += Ys[ty][k] * Ys[ty][k];
      sumXY += Xs[tx][k] * Ys[ty][k];
    }
    __syncthreads();
  }
  avgX = sumX / numCol;
  avgY = sumY / numCol;
  varX = (sumX2 - avgX*avgX*numCol) / (numCol-1);
  varY = (sumY2 - avgY*avgY*numCol) / (numCol-1);
  cov = (sumXY - avgX*avgY*numCol) / (numCol-1);
  rho = cov / sqrtf(varX*varY);
  outIdx = by*BlockSize*numRow + ty*numRow + bx*BlockSize + tx;
  output[outIdx] = rho;
}

int main(int argc, char *argv[]){
  int numRow = 16384;
  int numCol = 64;
  float *data, *cpuPD, *copyPD;
  float *gpuData, *gpuPD;
  time_t seed;
  int i, j;
  cudaEvent_t start;
  cudaEvent_t stop;
  float cpuT, gpuT;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  printf("numRow %d, numCol %d\n", numRow, numCol);

  data = (float *)malloc(numRow*numCol*sizeof(float));
  cpuPD = (float *)malloc(numRow*numRow*sizeof(float));
  copyPD = (float *)malloc(numRow*numRow*sizeof(float));
  time(&seed);
  srand48(seed);
  for (i=0; i<numRow; i++)
    for(j=0; j<numCol; j++)
      data[i*numCol + j] = drand48();

  cudaSetDevice(0);
  cudaMalloc((void**)&gpuData, numRow*numCol*sizeof(float));
  cudaMalloc((void**)&gpuPD, numRow*numRow*sizeof(float));

  //cpu pairwise distance
  cudaEventRecord(start, NULL);
  cpuPearson(data, numRow, numCol, cpuPD);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cpuT, start, stop);
  //sec
  printf("cpu time: %.2f sec\n", cpuT/1000);

  cudaEventRecord(start, NULL);
  cudaMemcpy(gpuData, data, numRow*numCol*sizeof(float),
	     cudaMemcpyHostToDevice);
  dim3 numThreads(BlockSize, BlockSize);
  dim3 numBlocks(numRow/BlockSize, numRow/BlockSize);
  gpuPearson<<<numBlocks, numThreads>>>(gpuData, numRow , numCol, gpuPD);
  cudaDeviceSynchronize();
  cudaMemcpy(copyPD, gpuPD, numRow*numRow*sizeof(float),
	     cudaMemcpyDeviceToHost);
  cudaEventRecord(stop, NULL);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpuT, start, stop);
  //sec, fold
  printf("gpu time: %.2f\n", gpuT/1000);
  printf("speedup: %.1f\n", cpuT/gpuT);
}
