#include <stdio.h>

__global__ void noop() {
    int i = threadIdx.x;
}

__global__ void no_div(int* data, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= len) {
        return;
    }
    data[idx] *= 2;
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

    if ((idx / warpSize) % 2 == 1) {
        data[idx] *= 3;
    } else {
        data[idx] *= 2;
    }
}

void run_kernel(int* ar, int length, char* name, void (*f)(int*, int)) {
    cudaEvent_t start, stop;
    float milliseconds = 0;

    dim3 grid, block;
    block = { 256 };
    grid = { (length + block.x - 1) / block.x };


    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    f<<<grid, block>>>(ar, length);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%s %f ms\n", name, milliseconds);
}


int main () {
    int length = 100000000;
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

    noop<<<10000, 10000>>>();

    run_kernel(ar1, length, "no_div", no_div);
    run_kernel(ar2, length, "diverge", diverge);
    run_kernel(ar3, length, "aligned_div", aligned_div);

    int *ar1_r = (int*) malloc(size);
    int *ar2_r = (int*) malloc(size);
    int *ar3_r = (int*) malloc(size);

    cudaMemcpy(ar1_r, ar1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ar2_r, ar2, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(ar3_r, ar3, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 50; ++i) {
        printf("%d %d %d\n", ar1_r[i], ar2_r[i], ar3_r[i]);
    }

    cudaFree(ar1);
    cudaFree(ar2);
    cudaFree(ar3);

    free(ar);
    free(ar1_r);
    free(ar2_r);
    free(ar3_r);
}