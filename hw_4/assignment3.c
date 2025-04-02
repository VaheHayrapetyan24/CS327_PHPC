#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 100000

void naive(int* a, int* res) {
	for (int i = 0; i < SIZE; ++i) {
		res[i] = 0;
		for (int j = 0; j <= i; ++j) {
			res[i] += a[j];
		}
	}
}


void simd(int* a, int* res) {
	int zeroes[8] = {0,0,0,0,0,0,0,0};
	__m256i zs = _mm256_loadu_epi32(zeroes);
	__m256i cr; // current result
	__m256i ca; // current a

	int crs[8]; // current stored result

	for (int i = 0; i < SIZE; ++i) {
		cr = zs; // copy the zeroes
		
		int j = 0;
		for (; j <= i - 8; j += 8) {
			ca = _mm256_loadu_epi32(&a[j]);
			cr = _mm256_add_epi32(cr, ca);
		}

		_mm256_storeu_epi32(crs, cr);

		res[i] = 0;
		for (; j <= i; ++j) {
			res[i] += a[j];
		}

		for (int k = 0; k < 8; ++k) {
			res[i] += crs[k];
		}
	}
}


int main() {
	srand(time(NULL));
	int a[SIZE];
	int n_res[SIZE];

	for (int i = 0; i < SIZE; ++i) {
		
		a[i] = rand() % 1000;
	}
	
	clock_t start = clock();
	naive(a, n_res);
	clock_t end = clock();

	printf("Naive Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);



	int simd_res[SIZE];

	start = clock();
	simd(a, simd_res);
	end = clock();


	printf("SIMD Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}
