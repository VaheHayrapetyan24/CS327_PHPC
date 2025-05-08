#include <stdio.h>
#include <sys/time.h>

#define XDIM 256
#define MUL 4

__host__ __device__ int get_idx(char v) {
    return (v == 'C') + (v == 'G') * 2 + (v == 'T') * 3;
}

__global__ void count(char* d_in, int* d_out, int n) {
    __shared__ int res[MUL][XDIM];

    int tid = threadIdx.x;
    char* cur_d_in = d_in + XDIM * blockIdx.x * MUL;

    res[0][tid] = 0;
    res[1][tid] = 0;
    res[2][tid] = 0;
    res[3][tid] = 0;

    res[get_idx(cur_d_in[tid * MUL])][tid]++;
    res[get_idx(cur_d_in[tid * MUL + 1])][tid]++;
    res[get_idx(cur_d_in[tid * MUL + 2])][tid]++;
    res[get_idx(cur_d_in[tid * MUL + 3])][tid]++;
    __syncthreads();

    for (int stride = XDIM / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            res[0][tid] += res[0][tid + stride];
            res[1][tid] += res[1][tid + stride];
            res[2][tid] += res[2][tid + stride];
            res[3][tid] += res[3][tid + stride];
        }

        __syncthreads();
    }


    if (tid == 0) {
        d_out[blockIdx.x * 4] = res[0][0];
        d_out[blockIdx.x * 4 + 1] = res[1][0];
        d_out[blockIdx.x * 4 + 2] = res[2][0];
        d_out[blockIdx.x * 4 + 3] = res[3][0];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main() {
    int N = 1 << 25;
    printf("N = %d\n", N);
    size_t in_size = sizeof(char) * N;
    char *h_in, *d_in;
    int *h_res, *d_res;

    dim3 grid, block;
    block.x = XDIM;
    grid.x = (N + block.x - 1) / (4 * block.x);
    printf("dimensions<<<%d, %d>>>\n", grid.x, block.x);

    size_t out_size = sizeof(int) * grid.x * 4;
    
    cudaError_t err_host = cudaMallocHost(&h_in, in_size);
    printf("err host: %s\n", cudaGetErrorString(err_host));


    h_res = (int*) malloc(out_size);

    cudaError_t err_in = cudaMalloc(&d_in, in_size);
    cudaError_t err_out = cudaMalloc(&d_res, out_size);

    printf("err1: %s, err2: %s\n", cudaGetErrorString(err_in), cudaGetErrorString(err_out));

    int r;
    for (int i = 0; i < N; ++i) {
        r = rand() % 4;
        switch (r)
        {
        case 0:
            h_in[i] = 'A';
            break;
        case 1:
            h_in[i] = 'C';
            break;
        case 2:
            h_in[i] = 'G';
            break;
        case 3:
            h_in[i] = 'T';
            break;
        default:
            break;
        }
    }

    double i_start, i_time;
    
    // STARTS HERE
    i_start = cpuSecond();
    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

    count<<<grid, block>>>(d_in, d_res, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_res, d_res, out_size, cudaMemcpyDeviceToHost);
    
    int device_sum[4] = {0,0,0,0};
    for (int i = 0; i < grid.x * 4; ++i) {
        device_sum[i % 4] += h_res[i];
    }

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    

    i_start = cpuSecond();
    int actual_res[4] = {0,0,0,0};
    for (int i = 0; i < N; ++i) {
        actual_res[get_idx(h_in[i])]++;
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);




    // printf("the string: ");
    // for (int i = 0; i < N; ++i) {
    //     printf("%c ", h_in[i]);
    // }
    printf("\n");

    printf("host results: ");
    for (int i = 0; i < 4; ++i) {
        printf("%d ", actual_res[i]);
    }
    printf("\n");

    printf("device results: ");
    for (int i = 0; i < 4; ++i) {
        printf("%d ", device_sum[i]);
    }
    printf("\n");

    free(h_res);
    cudaFree(h_in);
    cudaFree(d_in);
    cudaFree(h_res);
    cudaFree(d_res);
}