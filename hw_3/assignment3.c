#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>

#define THREADS_NUM 6
#define ITERATIONS 10
#define MAX_TEMPERATURE 100

double results[ITERATIONS];
double current_measurements[THREADS_NUM];

pthread_barrier_t barrier;
pthread_spinlock_t spinlock;


void* collect_data(void* arg) {
	int index = ((int*) arg);

	for (int i = 0; i < ITERATIONS; ++i) {
		current_measurements[index] = (double) (rand() % MAX_TEMPERATURE);

		printf("%d measured temperature %f \n", index, current_measurements[index]);

		pthread_barrier_wait(&barrier);

		// at this point either all of the threads could do the same calculcation since thechnically 
		// they all should get the same result
		// but I found this trylock method, which would help to easily decide which one will do it
		// otherwise I guess I could write something like - all threads choose a random number
		// whichever one has the highest will do the calculation

		if (pthread_spin_trylock(&spinlock) != 0) {
			printf("%d tried to lock, but was already locked\n", index);
			continue;
		}
		results[i] = 0;
		for (int j = 0; j < THREADS_NUM; ++j) {
			results[i] += current_measurements[j] / THREADS_NUM;
		}
		printf("%d locked and calculated average %f\n\n\n", index, results[i]);
		pthread_spin_unlock(&spinlock);
	}
	return NULL;
}

int main() {
	pthread_t threads[THREADS_NUM];

	pthread_spin_init(&spinlock, PTHREAD_PROCESS_PRIVATE);
	pthread_barrier_init(&barrier, NULL, THREADS_NUM);

	int thread_indices[THREADS_NUM];

	for (int i = 0; i < THREADS_NUM; ++i) {
		thread_indices[i] = i;
		if (pthread_create(&threads[i], NULL, &collect_data, (void*) thread_indices[i]) != 0)
			perror("Failed to create thread");
	}

	for (int i = 0; i < THREADS_NUM; ++i) {
		if (pthread_join(threads[i], NULL) != 0) 
			perror("Failed to join thread");
	}

	pthread_barrier_destroy(&barrier);
	pthread_spin_destroy(&spinlock);

	for (int i = 0; i < ITERATIONS; ++i) {
		printf("[%d]: %f ", i, results[i]);
	}
	printf("\n");
}
