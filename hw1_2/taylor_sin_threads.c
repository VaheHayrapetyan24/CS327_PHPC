#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>

#define NUM_OF_ITERATIONS 50000
#define NUM_OF_THREADS 10
#define X 0.3

struct thread_arguments {
	int thread_index;
	long double* sum;
};

long double next_term(double x, long double previous_term, int powr) {
	int original_powr = powr;
	while (powr > 0 && powr > original_powr - 2 * NUM_OF_THREADS) {
//		printf("powr %d, term %.30Lf\n", powr, previous_term);
		
		previous_term *= x/powr;
//		printf("asdf %.30Lf\n", previous_term);
		--powr;
	}

	long double result = (((original_powr - 1) / 2) % 2 ? -1 : 1) * fabs(previous_term);

//	printf("before asdf %.30Lf, %.30Lf, %d\n", previous_term, result, isinf(result));
	return isinf(result) ? 0.0L : result;
}


void* calculate_sum(void* args) {
	struct thread_arguments *cast_args = (struct thread_arguments*) args;
	printf("running in thread %d, sum is %d \n", cast_args -> thread_index, *(cast_args -> sum));


	int powr = 2*(cast_args -> thread_index) + 1;
	long double current = next_term(X, 1.0L, powr);
//	printf("first current %.30Lf\n", current);
	for (int i = cast_args -> thread_index; i < NUM_OF_ITERATIONS; i+= NUM_OF_THREADS) {
		*(cast_args -> sum) += current;
//		printf("summed %.30Lf, %.30Lf\n", *(cast_args -> sum), current);
		powr += 2 * NUM_OF_THREADS;
		current = next_term(X, current, powr);
	}


	return NULL;
}



int main() {
	printf("hello\n");
	
	long double sums[NUM_OF_THREADS] = {0.0L};
	pthread_t threads[NUM_OF_THREADS];

	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		struct thread_arguments *args = (struct thread_arguments*) malloc(sizeof(struct thread_arguments));
		args->thread_index = i;
		args->sum = &sums[i];
		
		int res = pthread_create(&(threads[i]), NULL, calculate_sum, (void*) args);
		if (res != 0) {
			printf("failed to create thread: %d\n", i);
			return -1;
		}	
	
	}


	long double sum = 0.0;
	for (int i = 0; i < NUM_OF_THREADS; ++i) {
		pthread_join(threads[i], NULL);
		sum += sums[i];
		printf("sum %.30Lf\n", sums[i]);
	}

	printf("sin(%f) = %.30Lf\n", X, sum); 



	
//	printf("%.30Lf\n", next_term(1, 1.0L/6, 13));


}
