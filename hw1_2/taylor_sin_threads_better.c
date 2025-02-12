#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>

#define NUM_OF_ITERATIONS 500000000
#define NUM_OF_THREADS 1
#define X 0.3L

struct thread_arguments {
	int thread_index;
	long double sum;

	// this one is used as return argument
	long double biggest_term;
};


void* calculate_sum(void* args) {
	struct thread_arguments *cast_args = (struct thread_arguments*) args;
//	printf("running in thread %d, sum is %Lf \n", cast_args -> thread_index, (cast_args -> sum));


	int start = cast_args -> thread_index * NUM_OF_ITERATIONS / NUM_OF_THREADS;
	int max = (cast_args -> thread_index + 1) * NUM_OF_ITERATIONS / NUM_OF_THREADS;
	long double current = X; // todo check if this works
				 //
//	printf("index %d, current %d, max %d, c %.30Lf\n", cast_args->thread_index, cast_args -> thread_index * NUM_OF_ITERATIONS / NUM_OF_THREADS, max, current);
	
	for (int i = start; i < max; ++i) {
		cast_args -> sum = (current - cast_args -> sum);
		current *= X * X / ((i + 1) * 2 * ((i + 1) * 2 + 1));
//		printf("cur res %.30Lf %d\n", X * X, i * 2 * (i * 2 + 1));
	}
	cast_args -> sum *= -1;

	cast_args -> biggest_term = current;



	return NULL;
}



int main() {
	clock_t start = clock();

	// long double sums[NUM_OF_THREADS] = {0.0L};
	pthread_t threads[NUM_OF_THREADS];
	struct thread_arguments* args[NUM_OF_THREADS];

	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		// I decided to have args array to be able to free the memory afterwards
		// Otherwise I would have to call free in calculate_sum, which seems wrong since
		// the args is created in this scope, so in this scope it should be deleted
		args[i] = (struct thread_arguments*) malloc(sizeof(struct thread_arguments));
		args[i]->thread_index = i;
		args[i]->sum = 0.0L;
		
		int res = pthread_create(&(threads[i]), NULL, calculate_sum, (void*) args[i]);
		if (res != 0) {
			printf("failed to create thread: %d\n", i);
			return -1;
		}	
	
	}


	long double sum = 0.0;
	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		pthread_join(threads[i], NULL);
//		printf("res %.30Lf\n", args[i] -> sum);

		sum += args[i]->sum * (i == 0 ? 1 : args[i - 1] -> biggest_term);
		free(args[i]);
//		printf("sum %.30Lf\n", sums[i]);
	}

	printf("sin(%Lf) = %.30Lf\n", X, sum); 



	printf("Threads: %d, Total iterations: %d, Time: %ld microseconds\n", NUM_OF_THREADS, NUM_OF_ITERATIONS, (int) clock() - start);
}
