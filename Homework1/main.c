#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS  3
#define TCOUNT 10
#define COUNT_LIMIT 12

int LIMIT_CONSUME = 1000;
int     count = 0;
typedef struct Element {
  struct Element *next;
} element_t;
pthread_mutex_t queue_mutex;
pthread_cond_t queue_cv;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

typedef struct ThreadData {
  element_t *queue;
  int number_consumed;
  int arrival_rate;
  int service_rate;
} thread_data_t;


void *consumer(void *thread_data) {

  pthread_mutex_lock(&count_mutex);
  int i = (*(thread_data_t *)thread_data).number_consumed;
  pthread_mutex_unlock(&count_mutex);
  int should_break = 0;
  while(i < LIMIT_CONSUME) {
    usleep(100*6);
    pthread_mutex_lock(&count_mutex);
    printf("consumer top, num: %d\n",(*(thread_data_t *)thread_data).number_consumed);
    element_t *queue = ((*(thread_data_t *)thread_data).queue);
    //printf("consumer queue addy: %p\n", queue);
    element_t *q_next = queue->next;
    if (q_next != NULL) {
      printf("consume the head...\n");
      queue->next = ((element_t *)(element_t *)(q_next)->next);
      free(q_next);
      (*(thread_data_t *)thread_data).number_consumed = ++i; 
    }
    if( (*(thread_data_t *)thread_data).number_consumed >= LIMIT_CONSUME) {
      pthread_cond_signal(&count_threshold_cv);
      printf("will break.\n");
      should_break = 1;
    }
    pthread_mutex_unlock(&count_mutex);
    if(should_break == 1) {
      break;
    }
  }
  pthread_exit(NULL);
}

void *producer(void *thread_data) {
  //printf("producer coming in\n");
  //sleep(1);
  pthread_mutex_lock(&count_mutex);
  int local_number_consumed = 0;
  local_number_consumed = (*(thread_data_t *) thread_data).number_consumed;
  pthread_mutex_unlock(&count_mutex);
  while (local_number_consumed < LIMIT_CONSUME) {
    usleep(100*4);
    pthread_mutex_lock(&count_mutex);
    element_t *queue = ((*(thread_data_t *)thread_data).queue);
    //printf("producer queue addy: %p\n", queue);
    element_t *cur_element = queue;
    int num_elements = 0;
    while (cur_element->next != NULL) {
      cur_element = cur_element->next;
      num_elements++;
    }
    // at the end of the queue, now
    element_t *new_end = (element_t *)malloc(sizeof(element_t));
    cur_element->next = new_end;
    new_end->next = NULL;
    local_number_consumed = (*(thread_data_t *) thread_data).number_consumed;
    if (local_number_consumed > LIMIT_CONSUME - 1) {
      printf("should be done!\n");
    }
    pthread_mutex_unlock(&count_mutex);
  }
  printf("out of producer loop\n");
  pthread_mutex_unlock(&count_mutex);
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  element_t *list_head = NULL;
  list_head = (element_t *)malloc(sizeof(element_t));
  list_head->next = NULL;
  int number_consumed = 0;
  thread_data_t *td = NULL;
  td = (thread_data_t *)malloc(sizeof(thread_data_t));
  td->number_consumed = 0;
  td->queue = list_head;
  td->arrival_rate = 3;
  td->service_rate = 4;
  int loop_idx = 0;
  /* For portability, explicitly create threads in a joinable state */
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_t consumer_thread;
  pthread_t producer_thread;
  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&count_mutex, NULL);
  pthread_cond_init(&count_threshold_cv, NULL);

  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  pthread_create(&producer_thread, &attr, producer, (void *)td);
  pthread_create(&consumer_thread, &attr, consumer, (void *)td);
  pthread_mutex_lock(&count_mutex);
  while (td->number_consumed < LIMIT_CONSUME) {
    pthread_cond_wait(&count_threshold_cv, &count_mutex);
  }
  pthread_mutex_unlock(&count_mutex);
  pthread_join(consumer_thread, NULL);
  pthread_join(producer_thread, NULL);
    
  pthread_exit(NULL);
}
