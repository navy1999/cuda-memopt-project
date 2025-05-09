#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void matmul_tiled(float *A, float *B, float *C, int n) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (n + TILE_SIZE -1)/TILE_SIZE; ++t) {
        int aCol = t*TILE_SIZE + threadIdx.x;
        int bRow = t*TILE_SIZE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row<n && aCol<n) ? A[row*n + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow<n && col<n) ? B[bRow*n + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row<n && col<n) C[row*n + col] = sum;
}

int main(int argc, char **argv) {
    if (argc!=2) { printf("Usage: %s <matrix_size>\n", argv[0]); return 1; }
    int N = atoi(argv[1]);
    const int trials=10; size_t bytes=N*N*sizeof(float);

    float *h_A=(float*)malloc(bytes), *h_B=(float*)malloc(bytes), *h_C=(float*)malloc(bytes);
    for(int i=0;i<N*N;++i){h_A[i]=1.0f; h_B[i]=2.0f;}

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,bytes); cudaMalloc(&d_B,bytes); cudaMalloc(&d_C,bytes);
    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE,TILE_SIZE), grid((N+TILE_SIZE-1)/TILE_SIZE,(N+TILE_SIZE-1)/TILE_SIZE);

    cudaEvent_t start,stop; cudaEventCreate(&start); cudaEventCreate(&stop);
    float total_ms=0;
    for(int t=0;t<trials;++t){
        cudaMemset(d_C,0,bytes);
        cudaEventRecord(start);
        matmul_tiled<<<grid,block>>>(d_A,d_B,d_C,N);
        cudaError_t err=cudaGetLastError();
        if(err!=cudaSuccess){printf("Kernel failed: %s\n",cudaGetErrorString(err));return 1;}
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms,start,stop);
        total_ms+=ms;
    }

    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);
    printf("[Tiled]  N=%d  AvgTime=%.3f ms\n", N, total_ms/trials);
    printf("Validation C[0]=%.1f\n", h_C[0]);
    fflush(stdout);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
