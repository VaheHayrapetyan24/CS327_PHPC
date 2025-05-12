#include <stdio.h>

#define BLOCKX 256
#define ITER 3


__global__ void gaussian_blur(float* input, float* output, int n) {
    int ind = blockDim.x * blockIdx.x + threadIdx.x + 1;

    if (ind > n) return;

    
    float val = (input[ind - 1] + input[ind + 1]) / 4 + input[ind] / 2;

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

    float *h_input, *h_output;
    float *d_input, *d_output;

    size_t size = sizeof(float) * N;
    cudaMallocHost(&h_input, size);
    cudaMallocHost(&h_output, size);

    for (int i = 0; i < N; ++i) {
        h_input[i] = (float) rand() / rand();
    }

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // On the first iteration this switches
    float *t1 = d_output, *t2 = d_input;
    for (int i = 0; i < ITER; ++i) {
        float *temp = t1;
        t1 = t2;
        t2 = temp;
        printf("%p %p\n", t1, t2);
        gaussian_blur<<<grid, block>>>(t1, t2, N - 1);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(h_output, t2, size, cudaMemcpyDeviceToHost);

    printf("input:\n");
    for (int i = 0; i < N; ++i) {
        printf(" %f", h_input[i]);
    }
    printf("\n");

    printf("output:\n");
    for (int i = 0; i < N; ++i) {
        printf(" %f", h_output[i]);
    }
    printf("\n");

    cudaFree(h_input);
    cudaFree(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

}