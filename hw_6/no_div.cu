#include <stdio.h>

__global__ void noop() {
    int i = threadIdx.x;
}

__global__ void no_div(int* data, int len) {
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    printf("idx %d\n", idx);
    if (idx < len) {
        data[idx] *= 2;
    }
}

__global__ void diverge(int* data, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) {
        return;
    }

    if (idx % 2) {
        data[idx] *= 3;
    } else {
        data[idx] *= 2;
    }
}

__global__ void aligned_div(int* data, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) {
        return;
    }

    if ((idx / warpSize) % 2) {
        data[idx] *= 3;
    } else {
        data[idx] *= 2;
    }
}


int main () {
    int length = 1000;
    int size = sizeof(int) * length;
    int* ar = (int*) malloc(size);

    for (int i = 0; i < length; ++i) {
        ar[i] = i;
    }

    int *ar1, *ar2, *ar3;

    cudaMalloc((void **) &ar1, size);
    cudaMalloc((void **) &ar2, size);
    cudaMalloc((void **) &ar3, size);

    cudaMemcpy(ar1, ar, size, cudaMemcpyHostToDevice);
    cudaMemcpy(ar2, ar1, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(ar3, ar1, size, cudaMemcpyDeviceToDevice);

    dim3 grid, block;
    block = { 256 };
    grid = { (length + block.x - 1) / block.x };


    noop<<<block, grid>>>();
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    float milliseconds = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);



    cudaEventRecord(start);
    no_div<<<grid, block>>>(ar1, length);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("no div %d ms\n", milliseconds);


}