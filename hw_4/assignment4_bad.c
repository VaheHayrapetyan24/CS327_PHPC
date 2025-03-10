#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 10000

void naive(int a[][SIZE], int* m, int* res) {
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			res[i] += a[i][j] * m[j];
		}
	}
}


void simd(int a[][SIZE], int* m, int* res) {
	int cr[8]; // current result
	__m256i cm; // current set of m's
	__m256i ca; // current set of row of a's

	int i = 0;
	for (; i <= SIZE - 8; i += 8) {
		cm = _mm256_loadu_epi32(&m[i]);

		for (int j = 0; j < SIZE; ++j) {
			ca = _mm256_loadu_epi32(&a[j][i]);

			// only taking the low 32 bits of the result, otherwise 64 bits is returned
			ca = _mm256_mullo_epi32(cm, ca); 

			_mm256_storeu_epi32(cr, ca);

			// THIS IS WHAT SLOWS EVERYTHING DOWN
			for (int k = 0; k < 8; ++k) {
				// made sure res is an array of 0s at the beginning
				res[j] += cr[k];
			}
		}
	}

	int previ = i;
	for (int j = 0; j < SIZE; ++j) {
		for (int i = previ; i < SIZE; ++i) {
			res[j] += a[j][i] * m[i];
		}
	}
}


int main() {
	srand(time(NULL));
	int a[SIZE][SIZE];
	int m[SIZE];
	int n_res[SIZE] = {0};

	for (int i = 0; i < SIZE; ++i) {
		m[i] = rand() % 1000;
		for (int j = 0; j < SIZE; ++j) {
			a[i][j] = rand() % 1000;
		}
	}
	
	clock_t start = clock();
	naive(a, m, n_res);
	clock_t end = clock();

	printf("Naive Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);



	int simd_res[SIZE] = {0};

	start = clock();
	simd(a, m, simd_res);
	end = clock();

	printf("SIMD Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}
