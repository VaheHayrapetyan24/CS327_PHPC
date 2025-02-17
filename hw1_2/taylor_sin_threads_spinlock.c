#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define NUM_OF_ITERATIONS 50000000
#define NUM_OF_THREADS 10
#define X 0.3L

struct thread_arguments {
	int thread_index;
};


long double sum = 0;
pthread_spinlock_t spinlock;

long double calculate_term(long int term) {
	long double res = X;
	for (long int i = 1; i <= term * 2 - 1; ++i) {
		res *= X / i;
	}
	return res;
}



void* calculate_sum(void* args) {
	struct thread_arguments *cast_args = (struct thread_arguments*) args;


	long int start = cast_args -> thread_index * NUM_OF_ITERATIONS / NUM_OF_THREADS;
	long int max = (cast_args -> thread_index + 1) * NUM_OF_ITERATIONS / NUM_OF_THREADS;
	long double current = calculate_term(start);
				 
	
	for (long int i = start + 1; i < max; ++i) {
		pthread_spin_lock(&spinlock);
		sum += current * ((i & 1) == 0 ? -1 : 1);
		pthread_spin_unlock(&spinlock);
		current *= X * X / ((i + 1) * 2 * ((i + 1) * 2 + 1));
	}


	return NULL;
}



int main() {
	struct timespec start, end;
	clock_gettime(CLOCK_REALTIME, &start);

	pthread_t threads[NUM_OF_THREADS];
	struct thread_arguments* args[NUM_OF_THREADS];


	pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);

	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		args[i] = (struct thread_arguments*) malloc(sizeof(struct thread_arguments));
		args[i]->thread_index = i;
		
		int res = pthread_create(&(threads[i]), NULL, calculate_sum, (void*) args[i]);
		if (res != 0) {
			printf("failed to create thread: %d\n", i);
			return -1;
		}	
	
	}


	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		pthread_join(threads[i], NULL);
		free(args[i]);
	}

	printf("sin(%Lf) = %.30Lf\n", X, sum);
	pthread_spin_destroy(&spinlock);	

	clock_gettime(CLOCK_REALTIME, &end);


	printf("Threads: %3d, Total iterations: %d, Time: %.0f nanoseconds\n", NUM_OF_THREADS, NUM_OF_ITERATIONS, (double) (end.tv_sec - start.tv_sec) * 10e9 + (double) (end.tv_nsec - start.tv_nsec));
}
