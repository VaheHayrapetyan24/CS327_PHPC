#include <stdio.h>
#include <immintrin.h>
#include <time.h>

#define SIZE 1000000000

#define W1 0.3f
#define W2 0.4f
#define W3 0.2f

void naive(float* a, float* res) {
	for (int i = 1; i < SIZE - 1; ++i) {
		res[i - 1] = a[i - 1] * W1 + a[i] * W2 + a[i + 1] * W3;
	}
}


void simd(float* a, float* res) {
	float w1a[8] = {W1, W1, W1, W1, W1, W1, W1, W1};
	float w2a[8] = {W2, W2, W2, W2, W2, W2, W2, W2};
	float w3a[8] = {W3, W3, W3, W3, W3, W3, W3, W3};


	__m256 w1 = _mm256_loadu_ps(w1a);
	__m256 w2 = _mm256_loadu_ps(w2a);
	__m256 w3 = _mm256_loadu_ps(w3a);

	__m256 curr_res;

	__m256 w1_res;
	__m256 w2_res;
	__m256 w3_res;
	int i;
	for (i = 0; i < SIZE - 10; i+= 8) { // up to SIZE - 10 becuase we're not going to calculate the last 2
		w1_res = _mm256_loadu_ps(&a[i]);
		w2_res = _mm256_loadu_ps(&a[i+1]);
		w3_res = _mm256_loadu_ps(&a[i+2]);
		w1_res = _mm256_mul_ps(w1, w1_res);
		w2_res = _mm256_mul_ps(w2, w2_res);
		w3_res = _mm256_mul_ps(w3, w3_res);

		w1_res = _mm256_add_ps(w1_res, w2_res);
		w1_res = _mm256_add_ps(w1_res, w3_res);
		_mm256_storeu_ps(&res[i], w1_res);
	}

	
	for (; i < SIZE - 2; ++i) {
		res[i] = W1 * a[i] + W2 * a[i + 1] + W3 * a[i + 2];
	}
}


int main() {
	srand(time(NULL));
        float a[SIZE];
	float n_res[SIZE - 2];

	for (int i = 0; i < SIZE; ++i) {
		a[i] = (float)rand();
	}

	
	clock_t start = clock();
	naive(a, n_res);
	clock_t end = clock();

	printf("Naive Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);



	float simd_res[SIZE - 2];

	start = clock();
	simd(a, simd_res);
	end = clock();


	printf("SIMD Time: %lf seconds\n", (double)(end - start) / CLOCKS_PER_SEC);
/*
	printf("naive: ");
	for (int i = 0; i < SIZE - 2; ++i) {
		printf("%f ", n_res[i]);
	}

	printf("\nsimd : ");
	for (int i = 0; i < SIZE - 2; ++i) {
                printf("%f ", simd_res[i]);
        }
	printf("\n");
*/

	
}
