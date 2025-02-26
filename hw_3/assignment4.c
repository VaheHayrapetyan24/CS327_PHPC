#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <pthread.h>

#define THREADS_NUM 6
#define MAX_WAIT_TIME 4

pthread_barrier_t barrier;


void stage() {
	sleep(rand() % MAX_WAIT_TIME);
}


void* start_game() {


	for (int i = 1; i < 4; ++i) {
		printf("%lu waiting for stage %d\n", pthread_self(), i);
		stage();
		pthread_barrier_wait(&barrier);
	}



	printf("%lu pipeline finished\n", pthread_self());
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
