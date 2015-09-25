#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS  3
#define TCOUNT 10
#define COUNT_LIMIT 12

int     count = 0;
typedef struct Element {
  int item;
  struct element_t *next;
} element_t;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;


int main(int argc, char *argv[]) {
  element_t *list_head = NULL;
  list_head = (element_t *)malloc(sizeof(element_t));
  pthread_exit(NULL);
}
