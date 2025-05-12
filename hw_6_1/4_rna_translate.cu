#include <stdio.h>
#include <sys/time.h>

#define XDIM 512
#define MUL 4

__constant__ char map[64];

__device__ __host__ char get_num(char i) {
    return (i == 'C') * 1 + (i == 'A') * 2 + (i == 'G') * 3;
}

__global__ void reduce(char* d_in, char* d_out, int n) {
    __shared__ char res[XDIM];
    __shared__ short counts[XDIM];

    int tid = threadIdx.x;
    char* cur_d_in = d_in + XDIM * blockIdx.x * MUL + tid * MUL;

    char cur_val = map[get_num(cur_d_in[0]) * 16 + get_num(cur_d_in[1]) * 4 + get_num(cur_d_in[2])];
    res[tid] = cur_val;
    counts[tid] = (cur_val != 0);
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            short right_count = counts[index + stride];
            memcpy(res + index + counts[index], res + index + stride, right_count);
            counts[index] += right_count;
        }
        __syncthreads();
    }

    short z_count = counts[0];
    if (tid < z_count) {
        d_out[blockDim.x * blockIdx.x + tid] = res[tid];
    }
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main() {
    char host_map[64] = {'F','F','L','L','S','S','S','S','Y','Y',0,0, 'C','C',0,'W','L','L','L','L','P','P','P','P','H','H','Q','Q','R','R','R','R','I','I','I','M','T','T','T','T','N','N','K','K','S','S','R','R','V','V','V','V','A','A','A','A','D','D','E','E','G','G','G','G'};
 
    cudaMemcpyToSymbol(map, host_map, sizeof(char) * 64);

    int N = 1 << 29;
    printf("N = %d\n", N);
    
    size_t in_size = sizeof(char) * N;
    char *h_in, *d_in, *original;
    char *h_res, *d_res, *actual_result;

    dim3 grid, block;
    block.x = XDIM;
    grid.x = (N + block.x - 1) / (MUL * block.x);
    printf("dimensions<<<%d, %d>>>\n", grid.x, block.x);

    size_t out_size = sizeof(char) * N / 4;
    
    original = (char*) malloc(in_size);
    actual_result = (char*) malloc(out_size);
    cudaError_t err_host_in = cudaMallocHost(&h_in, in_size);
    cudaError_t err_host_res = cudaMallocHost(&h_res, out_size);

    if (original == NULL || actual_result == NULL) {
        printf("failed to malloc\n");
    }

    printf("err host in: %s, err host res: %s\n", cudaGetErrorString(err_host_in), cudaGetErrorString(err_host_res));



    cudaError_t err_in = cudaMalloc(&d_in, in_size);
    cudaError_t err_out = cudaMalloc(&d_res, out_size);

    printf("err1: %s, err2: %s\n", cudaGetErrorString(err_in), cudaGetErrorString(err_out));

    // printf("string: ");
    int r;
    for (int i = 0; i < N; i += 4) {
        for (int j = i; j < i + 3; ++j) {
            r = rand() % 4;
            char v;
            switch (r)
            {
            case 0:
                v = 'A';
                break;
            case 1:
                v = 'C';
                break;
            case 2:
                v = 'G';
                break;
            case 3:
                v = 'U';
                break;
            default:
                break;
            }
    
            h_in[j] = v;
            original[j] = v;
            // printf("%c", v);
        }
        // printf(" ");
    }
    printf("\n");

    double i_start, i_time;
    
    // STARTS HERE
    i_start = cpuSecond();
    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

    reduce<<<grid, block>>>(d_in, d_res, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_res, d_res, out_size, cudaMemcpyDeviceToHost);
    
    // double device_sum = 0.0;
   

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    

    i_start = cpuSecond();
    for (int i = 0; i < N; i+=4) {
        char cur_val = host_map[get_num(original[i]) * 16 + get_num(original[i + 1]) * 4 + get_num(original[i + 2])];
        actual_result[i / 4] = cur_val;
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);

    printf("\n");

    // printf("host res:   ");
    // for (int i = 0; i < N / 4; ++i) {
    //     printf("%c", actual_result[i]);
    // }
    // printf("\n");

    // printf("device res: ");
    // for (int i = 0; i < N / 4; ++i) {
    //     printf("%c", h_res[i]);
    // }
    // printf("\n");

    cudaFree(h_in);
    cudaFree(d_in);
    cudaFree(h_res);
    cudaFree(d_res);
    free(original);
    free(actual_result);
}