#include <stdio.h>
#include <sys/time.h>

#define XDIM 512

__host__ __device__ char get_char(char v) {
    return (v == 'T') * 'U' + (v != 'T') * v;
}

__global__ void transcribe(char* d_in, char* d_out, int n) {
    int dimx = blockDim.x; 
    int ind = dimx * blockIdx.x * 4 + threadIdx.x;

    d_out[ind] = get_char(d_in[ind]);
    d_out[ind + dimx] = get_char(d_in[ind + dimx]);
    d_out[ind + 2 * dimx] = get_char(d_in[ind + 2 * dimx]);
    d_out[ind + 3 * dimx] = get_char(d_in[ind + 3 * dimx]);
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main() {
    int N = 1 << 28;
    printf("N = %d\n", N);
    size_t in_size = sizeof(char) * N;
    size_t out_size = in_size;

    char *h_in, *d_in;
    char *h_res, *d_res;

    dim3 grid, block;
    block.x = XDIM;
    grid.x = (N + block.x - 1) / (4 * block.x);
    printf("dimensions<<<%d, %d>>>\n", grid.x, block.x);

    
    cudaError_t err_host = cudaMallocHost(&h_in, in_size);
    printf("err host: %s\n", cudaGetErrorString(err_host));


    h_res = (char*) malloc(out_size);

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

    transcribe<<<grid, block>>>(d_in, d_res, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_res, d_res, out_size, cudaMemcpyDeviceToHost);

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    

    char* actual_res = (char*) malloc(out_size);
    i_start = cpuSecond();
    
    for (int i = 0; i < N; ++i) {
        actual_res[i] = get_char(h_in[i]);
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);




    // printf("the string: \n");
    // for (int i = 0; i < N; ++i) {
    //     printf("%c", h_in[i]);
    // }
    // printf("\n");

    // printf("host results: \n");
    // for (int i = 0; i < N; ++i) {
    //     printf("%c", actual_res[i]);
    // }
    // printf("\n");

    // printf("device results: \n");
    // for (int i = 0; i < N; ++i) {
    //     printf("%c", h_res[i]);
    // }
    // printf("\n");

    for (int i = 0; i < N; ++i) {
        if (actual_res[i] != h_res[i]) {
            printf("wrong results\n");
        }
    }

    free(h_res);
    cudaFree(h_in);
    cudaFree(d_in);
    cudaFree(h_res);
    cudaFree(d_res);
}