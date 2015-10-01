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
  double server_utilization;
  double mean_inter_arrival_time;
  double std_dev_arrival_time;
  double mean_inter_service_wait_time;
  double std_dev_service_wait_time;
  double mean_inter_service_time;
  double std_dev_service_time;
  double mean_queue_length;
  double std_dev_queue_length;
  int meta_data_memory;
} thread_data_t;


void *consumer(void *thread_data) {
  double random_value;
  int local_consumer = 0;
  double mean_customer_wait_time = 0;
  int number_of_waits = 0;
  pthread_mutex_lock(&count_mutex);
  thread_data_t *td = &(*(thread_data_t *)thread_data);
  local_consumer = td->consumers_that_have_started++;
  int i = td->number_consumed;
  pthread_mutex_unlock(&count_mutex);
  struct timespec abstime;
  abstime.tv_sec = 2;
  abstime.tv_nsec = 3000;

  struct timeval tv;
  gettimeofday(&tv,NULL);
  struct drand48_data buffer;
  srand48_r(tv.tv_sec+tv.tv_usec, &buffer);
  drand48_r(&buffer, &random_value);
  //thread_data_t td = (*(thread_data_t *)thread_data);
  random_value = -1 *(log(1.0-random_value)/td->service_rate);
  //printf("inter-service rate: %f\n", random_value);
  struct timespec sleepTime;
  sleepTime.tv_sec = (int)floor(random_value);
  sleepTime.tv_nsec = (random_value-sleepTime.tv_sec) * 1000000000L;

  double free_time = 0;
  double total_time = 0;
  while(i < LIMIT_CONSUME) {
    //printf("consumer waiting for lock, at 0, local: %d\n", local_consumer);
    struct timeval lock_begin, lock_end;
    gettimeofday(&lock_begin, NULL);
    pthread_mutex_lock(&count_mutex);
    gettimeofday(&lock_end, NULL);
    //printf("consumer, mutex lock 1 wait: %f\n", (lock_end.tv_sec - lock_begin.tv_sec + ((lock_end.tv_usec - lock_begin.tv_usec) / 1000000.0)));
    i = td->number_consumed;
    //printf("consumer top, num: %d\n", i);
    element_t *queue = td->queue;
    if (queue != NULL) {
      element_t *old_head = queue;
      td->queue = old_head->next;
      free(old_head);
      td->number_consumed = ++i; 
      pthread_mutex_unlock(&count_mutex);
      struct timeval tv1, tv2;
      gettimeofday(&tv1, NULL);

      nanosleep(&sleepTime, NULL);
      gettimeofday(&tv2, NULL);
      double wait_time = tv2.tv_sec - tv1.tv_sec + (tv2.tv_usec - tv1.tv_usec) / 1000000.0;
      mean_customer_wait_time += wait_time;
      mean_customer_wait_time /= 1.0 * ++number_of_waits;

      //printf("consumer waiting for lock at 1, local: %d\n", local_consumer);
      pthread_mutex_lock(&count_mutex);
      gettimeofday(&lock_end, NULL);
      total_time += (lock_end.tv_sec - lock_begin.tv_sec + ((lock_end.tv_usec - lock_begin.tv_usec) / 1000000.0));
      
      //printf("consumer received for lock at 1, local: %d\n", local_consumer);
    } else {
      // let's wait for the producer to inform us there is an item waiting for us.
      gettimeofday(&lock_begin, NULL);
      pthread_cond_timedwait(&count_threshold_cv, &count_mutex, &abstime);
      gettimeofday(&lock_end, NULL);
      free_time += (lock_end.tv_sec - lock_begin.tv_sec + ((lock_end.tv_usec - lock_begin.tv_usec) / 1000000.0));
      //printf("consumer, cond wait: %f\n", (lock_end.tv_sec - lock_begin.tv_sec + ((lock_end.tv_usec - lock_begin.tv_usec) / 1000000.0)));
    }
    i = td->number_consumed;
    if (i >= LIMIT_CONSUME) {
      td->consumers_that_have_finished += 1;
    }
    pthread_mutex_unlock(&count_mutex);
  }
  //printf("consumer exiting, local: %d\n", local_consumer);
  printf("mean service time: %f\n", mean_customer_wait_time);
  printf("utilization: %f\n", (total_time - free_time) / total_time);
  pthread_exit(NULL);
}

void *producer(void *thread_data) {
  //printf("producer coming in\n");
  //sleep(1);
  double mean_customer_wait_time = 0;
  int number_of_waits = 0;

  pthread_mutex_lock(&count_mutex);
  thread_data_t *td = (&(*(thread_data_t *) thread_data));
  int local_number_consumed = 0;
  local_number_consumed = td->number_consumed;
  pthread_mutex_unlock(&count_mutex);
  struct timeval tv1, tv2;

  struct timeval tv;
  gettimeofday(&tv,NULL);
  struct drand48_data buffer;
  srand48_r(tv.tv_sec+tv.tv_usec, &buffer);
  double random_value;
  drand48_r(&buffer, &random_value);
  random_value = -1 *(log(1.0-random_value)/td->arrival_rate);
  //printf("inter-arrival rate: %f\n", random_value);
  struct timespec sleepTime;
  sleepTime.tv_sec = (int)floor(random_value);
  sleepTime.tv_nsec = (random_value-sleepTime.tv_sec) * 1000000000L;
  while (local_number_consumed < LIMIT_CONSUME) {
    nanosleep(&sleepTime, NULL);
    //printf("producer sleep time: %f\n", (sleepTime.tv_sec + (sleepTime.tv_nsec / 1000000000.0)));

    gettimeofday(&tv1, NULL);
    struct timeval lock_begin, lock_end;
    gettimeofday(&lock_begin, NULL);
    pthread_mutex_lock(&count_mutex);
    gettimeofday(&lock_end, NULL);
    //printf("producer, lock after arrival, mutex lock wait: %f\n", (lock_end.tv_sec - lock_begin.tv_sec + ((lock_end.tv_usec - lock_begin.tv_usec) / 1000000.0)));
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
    }
    pthread_mutex_unlock(&count_mutex);
    gettimeofday(&tv2, NULL);
    double wait_time = (tv2.tv_sec - tv1.tv_sec + ((tv2.tv_usec - tv1.tv_usec) / 1000000.0));
    //printf("situ customer wait time: %f, rolling mean before: %f\n", wait_time, mean_customer_wait_time);
    mean_customer_wait_time += wait_time;
    mean_customer_wait_time /= 1.0 * ++number_of_waits;

  }
  //printf("out of producer loop\n");
  //printf("producer exiting, i: %d\n", local_number_consumed);
  printf("mean customer wait time: %f\n", mean_customer_wait_time);
  pthread_exit(NULL);
}

void *meta_data(void *thread_data) {
  pthread_mutex_lock(&count_mutex);
  thread_data_t *td = (&(*(thread_data_t *) thread_data));
  int use_meta_data_memory = td->meta_data_memory ? 1 : 0; // whether or not we should use the memory
  int mean_memory[td->meta_data_memory == 0 ? 1 :td->meta_data_memory];
  double number_of_collections = 0;
  double current_mean = 0;

  int cur_consumed = td->number_consumed;
  element_t *cur_node = td->queue;
  int queue_size = 0;
  while (cur_node != NULL) {
    cur_node = cur_node->next;
    queue_size++;
  }

  current_mean += queue_size;
  current_mean /= ++number_of_collections;
  
  pthread_mutex_unlock(&count_mutex);
  while (cur_consumed < LIMIT_CONSUME) {
    usleep(10000);
    //pthread_mutex_lock(&count_mutex);
    cur_consumed = td->number_consumed;
    cur_node = td->queue;
    queue_size = 0;
    while (cur_node != NULL) {
      cur_node = cur_node->next;
      queue_size++;
    }
    
    current_mean += queue_size;
    current_mean /= 1.0 * ++number_of_collections;
    //printf("Number processed: %d\n", cur_consumed);
    //pthread_mutex_unlock(&count_mutex);
  }
  printf("Queue mean: %f\n", current_mean);
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
		td->arrival_rate = (double)atof(argv[i]);
	      } else if (strcmp(argv[i-1], "-M") == 0 ) {
		td->service_rate = (double)atof(argv[i]);
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
  pthread_t md_thread;
  pthread_create(&md_thread, &attr, meta_data, (void *)td);
  i = 0;
  for (; i < number_consumers; i++) {
    //printf("joining consumer: %d\n", i);
    pthread_join(consumer_threads[i], NULL);
  }
  pthread_join(producer_thread, NULL);
  pthread_join(md_thread, NULL);
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
