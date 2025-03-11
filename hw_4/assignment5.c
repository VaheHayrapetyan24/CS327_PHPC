#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 500000000

void naive(char* a, char* res) {
	char c;
	for (int i = 0; i < SIZE; ++i) {
		c = a[i];
		res[i] = c - (c > 96 && c < 123) * 32;
	}
}


void simd(char* a, char* res) {
	__m256i gta;
	__m256i ltz;
	__m256i ca; // current a chunk
	__m256i ba; // bits to shift for a

	__m256i as = _mm256_set1_epi8(96); // vector of (a - 1)'s
	__m256i zs = _mm256_set1_epi8(123); // vector of (z + 1)'s

	__m256i tt = _mm256_set1_epi8(32); // thirty-twos

	int i = 0;
	for (; i <= SIZE - 32; i+= 32) {
		ca = _mm256_loadu_epi8(&a[i]);
		gta = _mm256_cmpgt_epi8(ca, as);
		ltz = _mm256_cmpgt_epi8(zs, ca);

		ba = _mm256_and_si256(gta, ltz);
		ba = _mm256_and_si256(ba, tt);

		ca = _mm256_xor_si256(ca, ba);

		_mm256_storeu_epi8(&res[i], ca);
	}


	char c;
	for (; i < SIZE; ++i) {
		c = a[i];
		res[i] = c - (c > 96 && c < 123) * 32;
	}
}


int main() {
	srand(time(NULL));

	char a[SIZE];
	char n_res[SIZE];

	for (int i = 0; i < SIZE; ++i) {
		a[i] = (65 + rand() % 57);
	}

	clock_t start = clock();
	naive(a, n_res);
	clock_t end = clock();

	printf("Naive Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);



	char simd_res[SIZE];

	start = clock();
	simd(a, simd_res);
	end = clock();

	printf("SIMD Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
}
