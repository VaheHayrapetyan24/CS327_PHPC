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

	float w1_res[SIZE - 2];
	int i;
	for (i = 0; i < SIZE - 10; i+= 8) { // up to SIZE - 10 becuase we're not going to calculate the last 2
		curr_res = _mm256_loadu_ps(&a[i]);
		curr_res = _mm256_mul_ps(w1, curr_res);
		_mm256_storeu_ps(&w1_res[i], curr_res);
	}

	for (; i < SIZE - 2; ++i) {
		w1_res[i] = a[i] * W1;
	}


	float w2_res[SIZE - 2];
	for (i = 1; i < SIZE - 9; i+= 8) {
                curr_res = _mm256_loadu_ps(&a[i]);
                curr_res = _mm256_mul_ps(w2, curr_res);
                _mm256_storeu_ps(&w2_res[i - 1], curr_res);
        }

	for (; i < SIZE - 1; ++i) {
                w2_res[i - 1] = a[i] * W2;
        }


	float w3_res[SIZE - 2];
        for (i = 2; i < SIZE - 8; i+= 8) {
                curr_res = _mm256_loadu_ps(&a[i]);
                curr_res = _mm256_mul_ps(w3, curr_res);
                _mm256_storeu_ps(&w3_res[i - 2], curr_res);
        }

	for (; i < SIZE; ++i) {
                w3_res[i - 2] = a[i] * W3;
        }

	for (i = 0; i <= SIZE - 11; i += 8) {
		__m256 w = _mm256_loadu_ps(&w1_res[i]);
		curr_res = _mm256_loadu_ps(&w2_res[i]);

		curr_res = _mm256_add_ps(w, curr_res);
		_mm256_storeu_ps(&res[i], curr_res);
	}

	for (i = 0; i <= SIZE - 11; i += 8) {
		__m256 w = _mm256_loadu_ps(&w3_res[i]);
		curr_res = _mm256_loadu_ps(&res[i]); // load already calculated addition of w1 and w2
                curr_res = _mm256_add_ps(w, curr_res); // use that and add it to w3
                _mm256_storeu_ps(&res[i], curr_res);
        }

	// since both of the past loops are iterating the same amount, we can use the latest i
	// to iterate the last few elements for both
	
	for (; i < SIZE - 2; ++i) {
		res[i] = w1_res[i] + w2_res[i] + w3_res[i];
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


	
}
