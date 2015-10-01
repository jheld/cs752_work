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
  double arrival_rate;
  double service_rate;
  int consumers_that_have_finished;
  int consumers_that_have_started;
} thread_data_t;


void *consumer(void *thread_data) {
  double random_value;
  int local_consumer = 0;
  pthread_mutex_lock(&count_mutex);
  thread_data_t *td = &(*(thread_data_t *)thread_data);
  local_consumer = td->consumers_that_have_started++;
  int i = td->number_consumed;
  pthread_mutex_unlock(&count_mutex);
  struct timespec abstime;
  abstime.tv_sec = 50;
  while(i < LIMIT_CONSUME) {
    //printf("consumer waiting for lock, at 0, local: %d\n", local_consumer);
    pthread_mutex_lock(&count_mutex);
    i = td->number_consumed;
    //printf("consumer top, num: %d\n", i);
    element_t *queue = td->queue;
    if (queue != NULL) {
      element_t *old_head = queue;
      td->queue = old_head->next;
      free(old_head);
      td->number_consumed = ++i; 
      pthread_mutex_unlock(&count_mutex);
      struct timeval tv;
      gettimeofday(&tv,NULL);
      struct drand48_data buffer;
      srand48_r(tv.tv_sec+tv.tv_usec, &buffer);
      drand48_r(&buffer, &random_value);
      //thread_data_t td = (*(thread_data_t *)thread_data);
      random_value = -1 *(log(1.0-random_value)/td->service_rate);
      //printf("inter-service rate: %f\n", random_value);
      usleep(random_value);
      //printf("consumer waiting for lock at 1, local: %d\n", local_consumer);
      pthread_mutex_lock(&count_mutex);
      //printf("consumer received for lock at 1, local: %d\n", local_consumer);
    } else {
      // let's wait for the producer to inform us there is an item waiting for us.
      //printf("consumer waiting for signal, local: %d\n", local_consumer);
      pthread_cond_timedwait(&count_threshold_cv, &count_mutex, &abstime);
    }
    i = td->number_consumed;
    if (i >= LIMIT_CONSUME) {
      td->consumers_that_have_finished += 1;
    }
    pthread_mutex_unlock(&count_mutex);
  }
  //printf("consumer exiting, local: %d\n", local_consumer);
  pthread_exit(NULL);
}

void *producer(void *thread_data) {
  //printf("producer coming in\n");
  //sleep(1);
  pthread_mutex_lock(&count_mutex);
  thread_data_t *td = (&(*(thread_data_t *) thread_data));
  int local_number_consumed = 0;
  local_number_consumed = td->number_consumed;
  pthread_mutex_unlock(&count_mutex);
  while (local_number_consumed < LIMIT_CONSUME) {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    struct drand48_data buffer;
    srand48_r(tv.tv_sec+tv.tv_usec, &buffer);
    double random_value;
    drand48_r(&buffer, &random_value);
    random_value = -1 *(log(1.0-random_value)/td->arrival_rate);
    //printf("inter-arrival rate: %f\n", random_value);
    usleep(random_value);
    pthread_mutex_lock(&count_mutex);
    local_number_consumed = td->number_consumed;
    element_t *queue = td->queue;
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
      td->queue = cur_element;
    }
    new_end->next = NULL;
    if (num_elements == 0) {
      // let the consumer know we have something for them.
      pthread_cond_signal(&count_threshold_cv);
    } else {
      //usleep(1000); // TODO: needs real mathz.
    }
    pthread_mutex_unlock(&count_mutex);
  }
  //printf("out of producer loop\n");
  //printf("producer exiting, i: %d\n", local_number_consumed);
  pthread_exit(NULL);
}

int main(int argc, char *argv[]) {
  element_t *list_head = NULL;
  //list_head = (element_t *)malloc(sizeof(element_t));
  //list_head->next = NULL;
  int i = 0;
  int number_consumers = 1;
  thread_data_t *td = NULL;
  td = (thread_data_t *)malloc(sizeof(thread_data_t));
  td->arrival_rate = 3;
  td->service_rate = 4;
  for (i = 1; i < argc; i++)  /* Skip argv[0] (program name). */
    {
      if (strcmp(argv[i], "-N") == 0 || 
	  strcmp(argv[i], "-L") == 0 || 
	  strcmp(argv[i], "-M") == 0 ||
	  strcmp(argv[i], "-C") == 0)  /* Process optional arguments. */
        {
	  /*
	   * The last argument is argv[argc-1].  Make sure there are
	   * enough arguments.
	   */

	  if (i + 1 <= argc - 1)  /* There are enough arguments in argv. */
            {
	      /*
	       * Increment 'i' twice so that you don't check these
	       * arguments the next time through the loop.
	       */

	      i++;
	      int intified = atoi(argv[i]); /* Convert string to int */
	      if (strcmp(argv[i-1], "-N") == 0) {
	        number_consumers = intified < 6 && intified > 0 ? intified : number_consumers;
	      } else if (strcmp(argv[i-1], "-L") == 0 ) {
		td->arrival_rate = (double)intified;
	      } else if (strcmp(argv[i-1], "-M") == 0 ) {
		td->service_rate = (double)intified;
	      } else {
		LIMIT_CONSUME = intified;
	      }
            }
        }
    }
  printf("num servers: %d, lambda: %f, mu: %f, cust: %d\n", number_consumers, td->arrival_rate, td->service_rate, LIMIT_CONSUME);
  int number_consumed = 0;
  td->number_consumed = 0;
  td->queue = NULL;//list_head;
  td->consumers_that_have_finished = 0;
  td->consumers_that_have_started = 0;
  int loop_idx = 0;
  /* For portability, explicitly create threads in a joinable state */
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_t consumer_thread;
  pthread_t producer_thread;
  /* Initialize mutex and condition variable objects */
  pthread_mutex_init(&count_mutex, NULL);
  pthread_cond_init(&count_threshold_cv, NULL);
  pthread_t consumer_threads[number_consumers];
  i = 0;
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
  for (; i < number_consumers; i++) {
    pthread_create(&consumer_threads[i], &attr, consumer, (void *)td);
  }
  pthread_create(&producer_thread, &attr, producer, (void *)td);
  pthread_mutex_lock(&count_mutex);
  number_consumed = td->number_consumed;
  pthread_mutex_unlock(&count_mutex);
  while (number_consumed < LIMIT_CONSUME) {
    usleep(10000);
    pthread_mutex_lock(&count_mutex);
    number_consumed = td->number_consumed;
    pthread_mutex_unlock(&count_mutex);
  }
  i = 0;
  for (; i < number_consumers; i++) {
    //printf("joining consumer: %d\n", i);
    pthread_join(consumer_threads[i], NULL);
  }
  pthread_join(producer_thread, NULL);
  element_t *queue = td->queue;
  // free the queue mem...
  while (queue != NULL) {
    element_t *old_head = queue;
    queue = queue->next;
    free(old_head);
  }
  free(td);
  /* Clean up and exit */
  pthread_attr_destroy(&attr);
  pthread_mutex_destroy(&count_mutex);
  pthread_cond_destroy(&count_threshold_cv);
  printf("finished joining consumers, total processed: %d\n", td->number_consumed);
  pthread_exit(NULL);
}
