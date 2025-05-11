#include <stdio.h>
#include <sys/time.h>

#define XDIM 256

__global__ void transcribe(char* d_in, int n) {
    int dimx = blockDim.x; 
    int ind = 4 * dimx * blockIdx.x + threadIdx.x;
    if (d_in[ind] == 'T') d_in[ind] = 'U';
    if (d_in[ind + dimx] == 'T') d_in[ind + dimx] = 'U';
    if (d_in[ind + 2 * dimx] == 'T') d_in[ind + 2 * dimx] = 'U';
    if (d_in[ind + 3 * dimx] == 'T') d_in[ind + 3 * dimx] = 'U';
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

int main() {
    int N = 1 << 29;
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

    transcribe<<<grid, block>>>(d_in, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_res, d_in, out_size, cudaMemcpyDeviceToHost);

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    char* actual_res = (char*) malloc(out_size);
    i_start = cpuSecond();
    
    for (int i = 0; i < N; ++i) {
        if(h_in[i] == 'T') h_in[i] = 'U';
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);

    for (int i = 0; i < N; ++i) {
        if (h_in[i] != h_res[i]) {
            printf("wrong results\n");
        }
    }

    free(h_res);
    cudaFree(h_in);
    cudaFree(d_in);
    cudaFree(h_res);
    cudaFree(d_res);
}