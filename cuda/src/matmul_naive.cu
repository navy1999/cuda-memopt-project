#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cuda_utils.h"

__global__ void matmul_naive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    if (N <= 0) { printf("Matrix size N must be positive.\n"); return 1; }
    const int trials = 10;
    size_t bytes = (size_t)N * N * sizeof(float);

    // Host allocations
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    for (int i = 0; i < N*N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaCheck(cudaMalloc(&d_A, bytes));
    cudaCheck(cudaMalloc(&d_B, bytes));
    cudaCheck(cudaMalloc(&d_C, bytes));
    cudaCheck(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 block(16,16), grid((N+15)/16,(N+15)/16);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float total_ms = 0;

    for (int t = 0; t < trials; ++t) {
        cudaCheck(cudaMemset(d_C, 0, bytes));
        cudaEventRecord(start);
        matmul_naive<<<grid,block>>>(d_A,d_B,d_C,N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            return 1;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    cudaCheck(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    printf("[Naive]  N=%d  AvgTime=%.3f ms\n", N, total_ms/trials);
    printf("Validation C[0]=%.1f\n", h_C[0]);
    fflush(stdout);

    // Cleanup
    cudaCheck(cudaEventDestroy(start));
    cudaCheck(cudaEventDestroy(stop));
    cudaCheck(cudaFree(d_A));
    cudaCheck(cudaFree(d_B));
    cudaCheck(cudaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return 0;
}
