#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>

#define THREADS_NUM 6
#define MAX_WAIT_TIME 10

pthread_barrier_t barrier;


void* start_game() {
	int wait_time = rand() % MAX_WAIT_TIME;

	printf("%lu will wait for %d seconds\n", pthread_self(), wait_time);

	sleep(wait_time);

	pthread_barrier_wait(&barrier);

	printf("%lu Game Started!\n", pthread_self());
	return NULL;
}

int main() {
	pthread_t threads[THREADS_NUM];

	pthread_barrier_init(&barrier, NULL, THREADS_NUM);

	for (int i = 0; i < THREADS_NUM; ++i) {
		if (pthread_create(&threads[i], NULL, &start_game, NULL) != 0)
			perror("Failed to create thread");
	}

	for (int i = 0; i < THREADS_NUM; ++i) {
		if (pthread_join(threads[i], NULL) != 0) 
			perror("Failed to join thread");
	}

	pthread_barrier_destroy(&barrier);
}
