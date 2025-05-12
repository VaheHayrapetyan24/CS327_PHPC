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

    char *original, *h_in, *d_in; // original is same as h_in at first. h_in is just pinned, and is used to write back the result too

    dim3 grid, block;
    block.x = XDIM;
    grid.x = (N + block.x - 1) / (4 * block.x);
    printf("dimensions<<<%d, %d>>>\n", grid.x, block.x);

    
    cudaError_t err_host = cudaMallocHost(&h_in, in_size);
    original = (char*) malloc(in_size);

    if (original == NULL) {
        printf("failed to alloc original");
    }
    printf("err host: %s\n", cudaGetErrorString(err_host));


    cudaError_t err_in = cudaMalloc(&d_in, in_size);

    printf("err1: %s\n", cudaGetErrorString(err_in));

    int r;
    for (int i = 0; i < N; ++i) {
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
            v = 'T';
            break;
        default:
            break;
        }

        h_in[i] = v;
        original[i] = v;
    }

    double i_start, i_time;
    
    // STARTS HERE
    i_start = cpuSecond();
    cudaMemcpy(d_in, h_in, in_size, cudaMemcpyHostToDevice);

    transcribe<<<grid, block>>>(d_in, N);

    cudaError_t err_sync = cudaDeviceSynchronize();
    printf("err sync: %s\n", cudaGetErrorString(err_sync));

    cudaMemcpy(h_in, d_in, in_size, cudaMemcpyDeviceToHost);

    i_time = cpuSecond() - i_start;

    printf("device took %f s\n", i_time);

    // ENDS HERE

    i_start = cpuSecond();
    
    for (int i = 0; i < N; ++i) {
        if(original[i] == 'T') original[i] = 'U';
    }
    i_time = cpuSecond() - i_start;

    printf("host took %f s\n", i_time);

    for (int i = 0; i < N; ++i) {
        if (h_in[i] != original[i]) {
            printf("wrong results\n");
        }
    }

    free(original);
    cudaFree(h_in);
    cudaFree(d_in);
}