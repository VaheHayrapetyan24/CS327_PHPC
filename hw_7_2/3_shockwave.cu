#include <stdio.h>

#define BLOCKX 256
#define ITER 20
#define C 0.02


__global__ void shockwave(float* prev, float* cur, float* next, int n) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x + 3;

    if (ind > n) return;

    float curr = cur[ind];
    float val = 2 * curr - prev[ind] + 
        C * (cur[ind - 3] - 6 * cur[ind - 2] + 15 * cur[ind - 1] - 20 * curr + 15 * cur[ind + 1] - 6 * cur[ind + 2] + cur[ind + 3]);
    printf("tid: %d, curr: %f, prev: %f, val: %f\n", ind, curr, prev[ind], val);

    __syncthreads();
    next[ind] = val;
}


void print_arr(float* arr, int n) {
    printf("\n");
    for (int i = 0; i < n; ++i) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main() {
    int N = (1 << 20) + 6;

    dim3 block, grid;
    block.x = BLOCKX;
    grid.x = (N + block.x - 7) / block.x;

    float *h_input, *h_output;
    float *d_1, *d_2, *d_3;

    size_t size = sizeof(float) * N;
    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_output, size);

    memset(h_input, 0, size);

    h_input[N / 2] = 100.0;

    cudaMalloc(&d_1, size);
    cudaMalloc(&d_2, size);
    cudaMalloc(&d_3, size);

    cudaMemcpy(d_1, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_2, d_1, size, cudaMemcpyDeviceToDevice);

    // On the first iteration this switches
    float *t1 = d_3, *t2 = d_1, *t3 = d_2;
    for (int i = 0; i < ITER; ++i) {
        float *temp = t1;
        t1 = t2;
        t2 = t3;
        t3 = temp;
        printf("%p %p %p\n", t1, t2, t3);
        shockwave<<<grid, block>>>(t1, t2, t3, N - 3);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_output, t3, size, cudaMemcpyDeviceToHost);

    // printf("input:\n");
    // for (int i = 0; i < N; ++i) {
    //     printf(" %f", h_input[i]);
    // }
    // printf("\n");

    // printf("output:\n");
    // for (int i = 0; i < N; ++i) {
    //     printf(" %f", h_output[i]);
    // }
    // printf("\n");

    cudaFree(h_input);
    cudaFree(h_output);
    cudaFree(d_1);
    cudaFree(d_2);
    cudaFree(d_3);
}