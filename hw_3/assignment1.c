#include <stdio.h>
#include <stdlib.h>

#include <pthread.h>

#define P 4
#define R 30

pthread_barrier_t barrier;
pthread_mutex_t mutex;

int current_max_die;
pthread_t max_winner;
int max_wins = 0;

void* toss_dice(void* args) {
	int my_wins = 0;
	for (int i = 0; i < R; ++i) {
		// Don't need to lock the max die here because
                // all of the threads will set it to 0 before reaching the first barrier
                current_max_die = 0;

		int die = rand() % 6 + 1;
		// At this point need to wait for all threads to toss their die
		pthread_barrier_wait(&barrier);

		printf("%lu: after first barrier current die %d\n", pthread_self(), die);	
		
		pthread_mutex_lock(&mutex);
		if (die > current_max_die) {
			current_max_die = die;
		}
		pthread_mutex_unlock(&mutex);
		printf("%lu: after first mutex unlock max die %d\n", pthread_self(), current_max_die);

		// I'm reusing the same barrier because the barrier number is equal to the threads count
		// so all of the threads are going to reach this line having the same state
		pthread_barrier_wait(&barrier);

		// No need to lock here because at this point no one will be updating the values of max variables
		
		printf("%lu: after second barrier current die %d max die %d\n", pthread_self(), die, current_max_die);
		if (die == current_max_die) {
			my_wins++;
			// only now need to lock
			// here will also reuse the same mutex because let's say 2 of P threads reach here simultaneously
			// they should not overwrite the max_wins, max_winner simultaneously
			// at the same time the rest of the threads will go to the first barrier and wait till these two finish
			// their business
			pthread_mutex_lock(&mutex);
	
			printf("%lu: my wins %d max wins %d \n", pthread_self(), my_wins, max_wins);
			if (my_wins > max_wins) {
				max_wins = my_wins;
				max_winner = pthread_self();
			}

			pthread_mutex_unlock(&mutex);
		}

		// need to wait at the end because otherwise we'll have a race condition of some thread setting current_max_die = 0 at the top
		// while others are still checking for die == current_max_die
		pthread_barrier_wait(&barrier);
		printf("\n");

	}
}


int main() {
	pthread_t threads[P];

	pthread_barrier_init(&barrier, NULL, P);
	pthread_mutex_init(&mutex, NULL);

	for (int i = 0; i < P; ++i) {
		if (pthread_create(&threads[i], NULL, &toss_dice, NULL) != 0) {
			perror("Failed to create thread");
		}
	}

	for (int i = 0; i < P; ++i) {
                if (pthread_join(threads[i], NULL) != 0) {
                        perror("Failed to join thread");
                }
        }

	printf("and the winner is %lu with %d wins\n", max_winner, max_wins);

	pthread_barrier_destroy(&barrier);
	pthread_mutex_destroy(&mutex);
}
