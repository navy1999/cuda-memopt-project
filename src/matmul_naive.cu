#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Matrix size N x N

__global__ void matmul_naive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k)
            sum += A[row * n + k] * B[k * n + col];
        C[row * n + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);
    float *A, *B, *C;
    float *d_A, *d_B, *d_C;

    // Allocate host memory
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    // Initialize matrices
    for (int i = 0; i < N*N; ++i) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+15)/16, (N+15)/16);

    // --- Timing Start ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    matmul_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    // --- Timing End ---

    // Calculate and print elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("[Naive] Kernel execution time: %.3f ms\n", milliseconds);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    printf("C[0]=%f\n", C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C);
    return 0;
}
