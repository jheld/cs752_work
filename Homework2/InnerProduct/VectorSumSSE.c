#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>

#include <xmmintrin.h>

int main(int argc, char *argv[]){
  unsigned int length = 4194304;
  int i, Size;
  float *a, *b, *c, *copyC;
  time_t seed;
  struct timeval tv1, tv2;
  float msecTotal;
  __m128 *aSSE, *bSSE, *cSSE;

  if (argc>1)
    sscanf(argv[1], "%d", &length);
  Size = sizeof(float) * length;
  posix_memalign((void **)&a, 16, Size);
  posix_memalign((void **)&b, 16, Size);
  posix_memalign((void **)&c, 16, Size);
  posix_memalign((void **)&copyC, 16, Size);
  time(&seed);
  srand48(seed);
  for (i=0; i<length; i++)
    a[i] = drand48(), b[i] = drand48();

  gettimeofday(&tv1, NULL);
  for (i=0; i<length; i++)
    c[i] = a[i] + b[i];
  gettimeofday(&tv2, NULL);
  printf("cpu time: %.3f msec\n", (float)(tv2.tv_sec-tv1.tv_sec)*1000 +
	 (float)(tv2.tv_usec-tv1.tv_usec)/1000.0);

  gettimeofday(&tv1, NULL);
  aSSE = (void*) a;
  bSSE = (void*) b;
  cSSE = (void*) copyC;
  for (i=0; i<length; i += 4, aSSE++, bSSE++, cSSE++)
    *cSSE = _mm_add_ps(*aSSE, *bSSE);
  gettimeofday(&tv2, NULL);
  printf("sse time: %.3f msec\n", (float)(tv2.tv_sec-tv1.tv_sec)*1000 +
	 (float)(tv2.tv_usec-tv1.tv_usec)/1000.0);

  for (i=0; i<length; i++)
    if (fabs(c[i]-copyC[i]) > 0.000001){
      printf("%d\t%f\t%f\n", i, c[i], copyC[i]);
      return 1;
    }
  return 0;
}
