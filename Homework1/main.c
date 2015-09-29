#include <pthread.h>
#include <math.h>
#include <time.h>
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
    usleep(100*3);
    pthread_mutex_lock(&count_mutex);
    printf("consumer top, num: %d\n",(*(thread_data_t *)thread_data).number_consumed);
    element_t *queue = ((*(thread_data_t *)thread_data).queue);
    if (queue != NULL) {
      element_t *old_head = queue;
      (*(thread_data_t *)thread_data).queue = old_head->next;
      free(old_head);
      (*(thread_data_t *)thread_data).number_consumed = ++i; 
      struct timeval tv;
      gettimeofday(&tv,NULL);
      struct drand48_data buffer;
      srand48_r(tv.tv_sec+tv.tv_usec, &buffer);
      double random_value;
      drand48_r(&buffer, &random_value);
      thread_data_t td = (*(thread_data_t *)thread_data);
      random_value = -1 *(log(1.0-random_value)/td.service_rate);
      //printf("inter-service rate: %f\n", random_value);

      usleep(random_value); // TODO: needs to be real mathz.
    } else {
      // let's wait for the producer to inform us there is an item waiting for us.
      //printf("about to wait for producer.\n");
      pthread_cond_wait(&count_threshold_cv, &count_mutex);
      //printf("producer gave go ahead.\n");
    }
    //printf("consumer queue addy: %p\n", queue);
    /*element_t *q_next = queue->next;
    if (q_next != NULL) {
      printf("consume the head...\n");
      queue->next = ((element_t *)(element_t *)(q_next)->next);
      free(q_next);
      (*(thread_data_t *)thread_data).number_consumed = ++i; 
      }*/
    if(i >= LIMIT_CONSUME) {
      //printf("will break.\n");
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
  thread_data_t td = (*(thread_data_t *) thread_data);
  local_number_consumed = (*(thread_data_t *) thread_data).number_consumed;
  pthread_mutex_unlock(&count_mutex);
  while (local_number_consumed < LIMIT_CONSUME) {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    struct drand48_data buffer;
    srand48_r(tv.tv_sec+tv.tv_usec, &buffer);
    double random_value;
    drand48_r(&buffer, &random_value);
    random_value = -1 *(log(1.0-random_value)/td.arrival_rate);
    //printf("inter-arrival rate: %f\n", random_value);
    usleep((int)random_value);
    pthread_mutex_lock(&count_mutex);
    element_t *queue = ((*(thread_data_t *)thread_data).queue);
    //printf("producer queue addy: %p\n", queue);
    element_t *cur_element = queue;
    int num_elements = 0;
    if (cur_element != NULL) {
      while (cur_element->next != NULL) {
	cur_element = cur_element->next;
	num_elements++;
      }
    }
    // at the end of the queue, now
    element_t *new_end = (element_t *)malloc(sizeof(element_t));
    if (cur_element != NULL) {
      cur_element->next = new_end;
    } else {
      cur_element = new_end;
      (*(thread_data_t *)thread_data).queue = cur_element;
    }
    new_end->next = NULL;
    local_number_consumed = (*(thread_data_t *) thread_data).number_consumed;
    if (num_elements == 0) {
      // let the consumer know we have something for them.
      //printf("about to give go ahead to consumer.\n");
      pthread_cond_signal(&count_threshold_cv);
      //printf("gave go ahead to consumer.\n");
    } else {
      usleep(1000); // TODO: needs real mathz.
    }
    /*    if (local_number_consumed > LIMIT_CONSUME - 1) {
      printf("should be done!\n");
      }*/
    pthread_mutex_unlock(&count_mutex);
  }
  //printf("out of producer loop\n");
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  element_t *list_head = NULL;
  //list_head = (element_t *)malloc(sizeof(element_t));
  //list_head->next = NULL;
  int number_consumed = 0;
  thread_data_t *td = NULL;
  td = (thread_data_t *)malloc(sizeof(thread_data_t));
  td->number_consumed = 0;
  td->queue = NULL;//list_head;
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
  number_consumed = td->number_consumed;
  pthread_mutex_unlock(&count_mutex);
  while (number_consumed < LIMIT_CONSUME) {
    usleep(10000);
    pthread_mutex_lock(&count_mutex);
    number_consumed = td->number_consumed;
    pthread_mutex_unlock(&count_mutex);
    //pthread_cond_wait(&count_threshold_cv, &count_mutex);
  }
  //printf("main thread finished waiting for consumption\n");
  //pthread_mutex_unlock(&count_mutex);
  pthread_join(consumer_thread, NULL);
  pthread_join(producer_thread, NULL);
  element_t *queue = td->queue;
  // free the queue mem...
  while (queue != NULL) {
    element_t *old_head = queue;
    queue = queue->next;
    free(old_head);
  }
  free(td);
  pthread_exit(NULL);
}
