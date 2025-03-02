#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 1000000000


int find_naive(int* a) {
	int min = *a;
	for (int i = 1; i < SIZE; ++i) {
		int curr = a[i];
		min = min < curr ? min : curr;
	}

	return min;
}

int find_simd(int * a) {
	__m256i vmin = _mm256_loadu_epi32(a);
	__m256i va;

	for (int i = 8; i <= SIZE - 8; ++i) {
		va = _mm256_loadu_epi32(&a[i]);
		
		vmin = _mm256_min_epi32(va, vmin);
	}

	va = _mm256_loadu_epi32(&a[SIZE-8]);
	vmin = _mm256_min_epi32(va, vmin);

	int res[8];
	_mm256_storeu_epi32(&res, vmin);

	/*
	for (int i = 0; i < 8; ++i) {
		printf("%d ", res[i]);
	}
	printf("\n");
	*/

	int min = res[0];
	for (int i = 1; i < 8; ++i) {
		min = min < res[i] ? min : res[i];
	}

	return min;
}

int main() {
	srand(time(NULL)); // seed the random
	int a[SIZE];


	for (int i = 0; i < SIZE; ++i) {
		a[i] = rand();
	//	 printf("%d ", a[i]);
	}	
	//printf("\n");

	clock_t start = clock();
	int naive = find_naive(a);
	clock_t end = clock();

	printf("naive min time: %lf seconds, result %d\n", (double)(end - start) / CLOCKS_PER_SEC, naive); 

	start = clock();
	int simd = find_simd(a);
	end = clock();

	printf("simd min time: %lf seconds, result %d\n", (double)(end - start) / CLOCKS_PER_SEC, simd);



}
