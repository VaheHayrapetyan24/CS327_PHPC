#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define NUM_OF_ITERATIONS 500000000
#define NUM_OF_THREADS 10
#define X 0.3

struct thread_arguments {
	int thread_index;
	long double* sum;
};

long double next_term(double x, long double previous_term, int powr) {
	int original_powr = powr;
	while (powr > 0 && powr > original_powr - 2 * NUM_OF_THREADS) {
		previous_term *= x/powr;
		--powr;
	}

	long double result = (((original_powr - 1) / 2) % 2 ? -1 : 1) * fabs(previous_term);

	return isinf(result) ? 0.0L : result;
}


void* calculate_sum(void* args) {
	struct thread_arguments *cast_args = (struct thread_arguments*) args;
//	printf("running in thread %d, sum is %d \n", cast_args -> thread_index, *(cast_args -> sum));


	int powr = 2*(cast_args -> thread_index) + 1;
	long double current = next_term(X, 1.0L, powr);
	for (int i = cast_args -> thread_index; i < NUM_OF_ITERATIONS; i+= NUM_OF_THREADS) {
		*(cast_args -> sum) += current;
		powr += 2 * NUM_OF_THREADS;
		current = next_term(X, current, powr);
	}


	return NULL;
}



int main() {
	clock_t start = clock();

	long double sums[NUM_OF_THREADS] = {0.0L};
	pthread_t threads[NUM_OF_THREADS];
	struct thread_arguments* args[NUM_OF_THREADS];

	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		// I decided to have args array to be able to free the memory afterwards
		// Otherwise I would have to call free in calculate_sum, which seems wrong since
		// the args is created in this scope, so in this scope it should be deleted
		args[i] = (struct thread_arguments*) malloc(sizeof(struct thread_arguments));
		args[i]->thread_index = i;
		args[i]->sum = &sums[i];
		
		int res = pthread_create(&(threads[i]), NULL, calculate_sum, (void*) args[i]);
		if (res != 0) {
			printf("failed to create thread: %d\n", i);
			return -1;
		}	
	
	}


	long double sum = 0.0;
	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		pthread_join(threads[i], NULL);
		sum += sums[i];
		free(args[i]);
//		printf("sum %.30Lf\n", sums[i]);
	}

	printf("sin(%f) = %.30Lf\n", X, sum); 



	printf("Threads: %d, Total iterations: %d, Time: %ld microseconds\n", NUM_OF_THREADS, NUM_OF_ITERATIONS, (int) clock() - start);
}
