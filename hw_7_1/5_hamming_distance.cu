#include <stdio.h>
#include <sys/time.h>

#define XDIM 1024
#define MUL 4

__global__ void count(char* d_1_in, char* d_2_in, int* d_out, int n) {
    __shared__ int res[XDIM];

    int tid = threadIdx.x;
    char* cur_d_1_in = d_1_in + XDIM * blockIdx.x * MUL;
    char* cur_d_2_in = d_2_in + XDIM * blockIdx.x * MUL;

    int ftid = tid * 4;
    res[tid] = (cur_d_1_in[ftid] != cur_d_2_in[ftid]);
    res[tid] += (cur_d_1_in[ftid + 1] != cur_d_2_in[ftid + 1]);
    res[tid] += (cur_d_1_in[ftid + 2] != cur_d_2_in[ftid + 2]);
    res[tid] += (cur_d_1_in[ftid + 3] != cur_d_2_in[ftid + 3]);
    __syncthreads();

    for (int stride = XDIM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            res[tid] += res[tid + stride];
        }

        __syncthreads();
    }


    if (tid == 0) {
        d_out[blockIdx.x] = res[0];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

char get_rand() {
    switch (rand() % 4)
    {
    case 0:
        return 'A';
    case 1:
        return 'C';
    case 2:
        return 'G';
    case 3:
        return 'T';
    }
}

int main() {
    int N = 1 << 29;
    printf("N = %d\n", N);
    size_t in_size = sizeof(char) * N;
    char *h_1_in, *h_2_in;
    char *d_1_in, *d_2_in;
    int *d_res, *h_res, actual_res = 0;

    dim3 grid, block;
    block.x = XDIM;
    grid.x = (N + block.x - 1) / (4 * block.x);
    printf("dimensions<<<%d, %d>>>\n", grid.x, block.x);

    size_t out_size = sizeof(int) * grid.x;
    
    cudaError_t err_1_host = cudaMallocHost(&h_1_in, in_size);
    cudaError_t err_2_host = cudaMallocHost(&h_2_in, in_size);
    cudaError_t err_3_host = cudaMallocHost(&h_res, out_size);
    printf("err host: 1: %s 2: %s 3: %s\n", cudaGetErrorString(err_1_host), cudaGetErrorString(err_2_host), cudaGetErrorString(err_3_host));

    cudaError_t err_1_in = cudaMalloc(&d_1_in, in_size);
    cudaError_t err_2_in = cudaMalloc(&d_2_in, in_size);
    cudaError_t err_out = cudaMalloc(&d_res, out_size);

    printf("err1: %s, err2: %s, out: %s\n", cudaGetErrorString(err_1_in), cudaGetErrorString(err_2_in), cudaGetErrorString(err_out));

    for (int i = 0; i < N; ++i) {
        h_1_in[i] = get_rand();
        h_2_in[i] = get_rand();
    }

    double i_start, i_time;
    
    // STARTS HERE
    i_start = cpuSecond();
    cudaMemcpy(d_1_in, h_1_in, in_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_2_in, h_2_in, in_size, cudaMemcpyHostToDevice);

    count<<<grid, block>>>(d_1_in, d_2_in, d_res, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_res, d_res, out_size, cudaMemcpyDeviceToHost);
    
    int device_sum = 0;
    for (int i = 0; i < grid.x; ++i) {
        device_sum += h_res[i];
    }

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    

    i_start = cpuSecond();

    for (int i = 0; i < N; ++i) {
        actual_res += (h_1_in[i] != h_2_in[i]);
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);


    printf("\n");

    printf("results: host: %d, device: %d\n",  actual_res, device_sum);

    cudaFree(h_1_in);
    cudaFree(h_2_in);
    cudaFree(d_1_in);
    cudaFree(d_2_in);
    cudaFree(h_res);
    cudaFree(d_res);
}