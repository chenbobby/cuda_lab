#include <cstdio>
#include <time.h>

#define N (10 * 1000 * 1000)

__global__ void vec_add_kernel(float* out, float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void vec_add_gpu(float* out_h, float* a_h, float* b_h, int n) {}

int main() {
    // Initialize input vectors on host.
    float* a_h = (float*)malloc(sizeof(float) * N);
    float* b_h = (float*)malloc(sizeof(float) * N);

    // Initialize input values.
    for (int i = 0; i < N; i++) {
        a_h[i] = 3.0f;
        b_h[i] = 2.0f;
    }

    // Initialize input vectors on device.
    float* a_d;
    float* b_d;

    cudaMalloc(&a_d, sizeof(float) * N);
    cudaMalloc(&b_d, sizeof(float) * N);

    // Copy input values from host to device memory.
    cudaMemcpy(a_d, a_h, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Initialize output vector on host.
    float* out_h = (float*)malloc(sizeof(float) * N);

    // Initialize output vector on device.
    float* out_d;

    cudaMalloc(&out_d, sizeof(float) * N);

    // Perform vector addition using GPU.
    struct timespec start, end;
    dim3 gridDim = (N + 255) / 256;
    dim3 blockDim = 256;

    clock_gettime(CLOCK_MONOTONIC, &start);
    vec_add_kernel<<<gridDim, blockDim>>>(out_d, a_d, b_d, N);
    clock_gettime(CLOCK_MONOTONIC, &end);

    long delta =
        (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec);
    printf("vec_add_gpu time: %.10lins\n", delta);

    // Copy output vector from device to host memory.
    cudaMemcpy(out_h, out_d, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // Assert the output is correct.
    for (int i = 0; i < N; i++) {
        if (out_h[i] != 5.0f) {
            printf("Error: out_h[%d] = %f\n", i, out_h[i]);
            return 1;
        }
    }

    // Free device memory.
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);

    return 0;
}
