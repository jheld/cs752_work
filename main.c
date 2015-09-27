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

typedef struct ThreadData {
  element_t *queue;
  int number_consumed;
} thread_data_t;


void *consumer(void *thread_data) {

  int i = 0;
  pthread_mutex_lock(&count_mutex);
  for(; i < 989999900; i++) {
    
  }
  printf("finished the loop\n");
  (*(thread_data_t *)thread_data).number_consumed = 45001;
  pthread_cond_signal(&count_threshold_cv);
  pthread_mutex_unlock(&count_mutex);
  pthread_exit(NULL);
}

void *producer(void *thread_data) {
  printf("producer coming in\n");
  sleep(1);
  pthread_mutex_lock(&count_mutex);
  while ((*(thread_data_t *) thread_data).number_consumed < 45000) {
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
  }
  printf("final count: %d\n", (*(thread_data_t *) thread_data).number_consumed);
  pthread_mutex_unlock(&count_mutex);
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  element_t *list_head = NULL;
  list_head = (element_t *)malloc(sizeof(element_t));
  int number_consumed = 0;
  thread_data_t *td = NULL;
  td = (thread_data_t *)malloc(sizeof(thread_data_t));
  td->number_consumed = 0;
  td->queue = list_head;
  int loop_idx = 0;
  /* For portability, explicitly create threads in a joinable state */
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_t consumer_thread;
  pthread_t producer_thread;
  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&count_mutex, NULL);
  pthread_cond_init (&count_threshold_cv, NULL);

  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&consumer_thread, &attr, consumer, (void *)td);
  pthread_create(&producer_thread, &attr, producer, (void *)td);
  pthread_join(consumer_thread, NULL);
  pthread_join(producer_thread, NULL);
    
  pthread_exit(NULL);
}
