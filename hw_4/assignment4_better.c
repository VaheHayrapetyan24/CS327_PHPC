#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 50000

void naive(int a[][SIZE], int* m, int* res) {
	for (int i = 0; i < SIZE; ++i) {
		for (int j = 0; j < SIZE; ++j) {
			res[i] += a[i][j] * m[j];
		}
	}
}


void simd(int a[][SIZE], int* m, int* res) {
	int zeroes[8] = {0,0,0,0,0,0,0,0};
	__m256i zs = _mm256_loadu_epi32(zeroes);


	__m256i cr; // current result
	__m256i cm; // current set of m's
	__m256i ca; // current set of row of a's

	int r[8];

	int j;
	for (int i = 0; i < SIZE; ++i) {
		cr = zs;
		for (j = 0; j <= SIZE - 8; j += 8) {
			cm = _mm256_loadu_epi32(&m[j]);
			ca = _mm256_loadu_epi32(&a[i][j]);

			// only taking the low 32 bits of the result, otherwise 64 bits is returned
			ca = _mm256_mullo_epi32(cm, ca); 
			cr = _mm256_add_epi32(cr, ca);

		}


		_mm256_storeu_epi32(r, cr);

		for (int k = 0; k < 8; ++k) {
			// made sure res is an array of 0s at the beginning
			res[i] += r[k];
		}
	}

	int prevj = j;
	for (int i = 0; i < SIZE; ++i) {
		for (j = prevj; j < SIZE; ++j) {
			res[i] += a[i][j] * m[j];
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
