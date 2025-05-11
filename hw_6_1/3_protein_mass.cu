#include <stdio.h>
#include <sys/time.h>

#define XDIM 256
#define MUL 4

__constant__ float weights[25];

__global__ void reduce(char* d_in, float* d_out, int n) {
    __shared__ float res[XDIM];

    int tid = threadIdx.x;
    char* cur_d_in = d_in + XDIM * blockIdx.x * MUL;

    res[tid] = weights[cur_d_in[tid * MUL] - 'A'];
    res[tid] += weights[cur_d_in[tid * MUL + 1] - 'A'];
    res[tid] += weights[cur_d_in[tid * MUL + 2] - 'A'];
    res[tid] += weights[cur_d_in[tid * MUL + 3] - 'A'];
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

char get_char_by_i(int i) {
    if (i == 0) {
        return 'G';
    }
    if (i == 1) {
        return 'A';
    }
    if (i == 2) {
        return 'S';
    }
    if (i == 3) {
        return 'P';
    }
    if (i == 4) {
        return 'V';
    }
    if (i == 5) {
        return 'T';
    }
    if (i == 6) {
        return 'C';
    }
    if (i == 7) {
        return 'I';
    }
    if (i == 8) {
        return 'L';
    }
    if (i == 9) {
        return 'N';
    }
    if (i == 10) {
        return 'D';
    }
    if (i == 11) {
        return 'Q';
    }
    if (i == 12) {
        return 'K';
    }
    if (i == 13) {
        return 'E';
    }
    if (i == 14) {
        return 'M';
    }
    if (i == 15) {
        return 'H';
    }
    if (i == 16) {
        return 'F';
    }
    if (i == 17) {
        return 'R';
    }
    if (i == 18) {
        return 'Y';
    }
    if (i == 19) {
        return 'W';
    }
} 

int main() {
    float host_weights[25];

    host_weights['A' - 'A'] = 71.04;
    host_weights['C' - 'A'] = 103.31;
    host_weights['D' - 'A'] = 115.03;
    host_weights['E' - 'A'] = 129.04;
    host_weights['F' - 'A'] = 147.07;
    host_weights['G' - 'A'] = 57.02;
    host_weights['H' - 'A'] = 137.06;
    host_weights['I' - 'A'] = 113.08;
    host_weights['K' - 'A'] = 128.09;
    host_weights['L' - 'A'] = 113.08;
    host_weights['M' - 'A'] = 131.04;
    host_weights['N' - 'A'] = 114.04;
    host_weights['P' - 'A'] = 97.05;
    host_weights['Q' - 'A'] = 128.06;
    host_weights['R' - 'A'] = 156.1;
    host_weights['S' - 'A'] = 87.03;
    host_weights['T' - 'A'] = 101.05;
    host_weights['V' - 'A'] = 99.07;
    host_weights['W' - 'A'] = 186.08;
    host_weights['Y' - 'A'] = 163.06;

    for (int i = 0; i < 25; ++i) {
        printf("%f ", host_weights[i]);
    }
    printf("\n");
 
    cudaMemcpyToSymbol(weights, host_weights, sizeof(float) * 25);

    int N = 1 << 29;
    printf("N = %d\n", N);
    
    size_t in_size = sizeof(char) * N;
    char *h_in, *d_in;
    float *h_res, *d_res;

    dim3 grid, block;
    block.x = XDIM;
    grid.x = (N + block.x - 1) / (MUL * block.x);
    printf("dimensions<<<%d, %d>>>\n", grid.x, block.x);

    size_t out_size = sizeof(float) * grid.x;
    
    cudaError_t err_host_in = cudaMallocHost(&h_in, in_size);
    cudaError_t err_host_res = cudaMallocHost(&h_res, out_size);

    printf("err host in: %s, err host res: %s\n", cudaGetErrorString(err_host_in), cudaGetErrorString(err_host_res));



    cudaError_t err_in = cudaMalloc(&d_in, in_size);
    cudaError_t err_out = cudaMalloc(&d_res, out_size);

    printf("err1: %s, err2: %s\n", cudaGetErrorString(err_in), cudaGetErrorString(err_out));

    int r;
    for (int i = 0; i < N; ++i) {
        r = rand() % 20;
        h_in[i] = get_char_by_i(r);
    }


    double i_start, i_time;
    
    // STARTS HERE
    i_start = cpuSecond();
    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

    reduce<<<grid, block>>>(d_in, d_res, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_res, d_res, out_size, cudaMemcpyDeviceToHost);
    
    double device_sum = 0.0;
    for (int i = 0; i < grid.x; ++i) {
        device_sum += h_res[i];
    }

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    

    i_start = cpuSecond();
    double host_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        host_sum += host_weights[h_in[i] - 'A'];
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);

    printf("\n");

    printf("host results  : %f \n", host_sum);

    printf("device results: %f \n", device_sum);

    cudaFree(h_in);
    cudaFree(d_in);
    cudaFree(h_res);
    cudaFree(d_res);
}