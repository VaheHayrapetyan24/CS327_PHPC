#include <stdio.h>

#define BLOCKX 256
#define ITER 10


__global__ void median_filter(int* input, int* output, int n) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (ind > n) return;

    
    int a = input[ind - 1], b = input[ind], c = input[ind + 1];
    float val = a + b + c - min(min(a, b), c) - max(max(a, b), c);

    __syncthreads();
    output[ind] = val;
}


void print_arr(float* arr, int n) {
    printf("\n");
    for (int i = 0; i < n; ++i) {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

int main() {
    int N = (1 << 20) + 2;

    dim3 block, grid;
    block.x = BLOCKX;
    grid.x = (N + block.x - 3) / block.x;

    int *h_input, *h_output;
    int *d_input, *d_output;

    size_t size = sizeof(int) * N;
    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_output, size);

    for (int i = 0; i < N; ++i) {
        h_input[i] = rand() % 1000;
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // On the first iteration this switches
    int *t1 = d_output, *t2 = d_input;
    for (int i = 0; i < ITER; ++i) {
        int *temp = t1;
        t1 = t2;
        t2 = temp;
        printf("%p %p\n", t1, t2);
        median_filter<<<grid, block>>>(t1, t2, N - 1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_output, t2, size, cudaMemcpyDeviceToHost);

    printf("input:\n");
    for (int i = 0; i < N; ++i) {
        printf(" %d", h_input[i]);
    }
    printf("\n");

    printf("output:\n");
    for (int i = 0; i < N; ++i) {
        printf(" %d", h_output[i]);
    }
    printf("\n");

    cudaFree(h_input);
    cudaFree(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

}