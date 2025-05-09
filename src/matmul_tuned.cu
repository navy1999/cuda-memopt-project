#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
typedef float4 vec4;

__global__ void matmul_tuned(float *A, float *B, float *C, int n) {
    __shared__ vec4 As[TILE_SIZE][TILE_SIZE/4 + 1];
    __shared__ vec4 Bs[TILE_SIZE][TILE_SIZE/4 + 1];

    int row = blockIdx.y*TILE_SIZE + threadIdx.y;
    int col = blockIdx.x*TILE_SIZE + threadIdx.x;
    float sum = 0.0f;
    int iterations = (n + TILE_SIZE -1)/TILE_SIZE;

    for(int t=0;t<iterations;++t){
        int aColBase=t*TILE_SIZE + threadIdx.x - (threadIdx.x%4);
        int bRowBase=t*TILE_SIZE + threadIdx.y - (threadIdx.y%4);
        int vecIdx = threadIdx.x/4;

        if(row<n && aColBase<n) As[threadIdx.y][vecIdx] = ((vec4*)A)[row*(n/4)+aColBase/4];
        else As[threadIdx.y][vecIdx]=make_float4(0,0,0,0);

        if(bRowBase<n && col<n) Bs[threadIdx.y][vecIdx] = ((vec4*)B)[bRowBase*(n/4)+col/4];
        else Bs[threadIdx.y][vecIdx]=make_float4(0,0,0,0);

        __syncthreads();
        #pragma unroll
        for(int k=0;k<TILE_SIZE/4;++k){
            vec4 a=As[threadIdx.y][k], b=Bs[k][threadIdx.x/4];
            sum += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
        __syncthreads();
    }
    if(row<n && col<n) C[row*n + col] = sum;
}

int main(int argc,char**argv){
    if(argc!=2){printf("Usage: %s <matrix_size>\n",argv[0]);return 1;}
    int N=atoi(argv[1]); if(N%TILE_SIZE){printf("N must be multiple of %d\n",TILE_SIZE);return 1;}
    const int trials=10; size_t bytes=N*N*sizeof(float);

    float *h_A=(float*)malloc(bytes),*h_B=(float*)malloc(bytes),*h_C=(float*)malloc(bytes);
    for(int i=0;i<N*N;++i){h_A[i]=1.0f;h_B[i]=2.0f;}
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
        matmul_tuned<<<grid,block>>>(d_A,d_B,d_C,N);
        cudaError_t e=cudaGetLastError();
        if(e!=cudaSuccess){printf("Kernel failed: %s\n",cudaGetErrorString(e));return 1;}
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms,start,stop); total_ms+=ms;
    }
    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);

    printf("[Tuned]  N=%d  AvgTime=%.3f ms\n",N,total_ms/trials);
    printf("Validation C[0]=%.1f\n",h_C[0]); fflush(stdout);

    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
