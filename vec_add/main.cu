#include <cstdio>

#define N 10485760 // 10 MiB

void vec_add_cpu(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

__global__ void vec_add_kernel(float* out, float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

void vec_add_gpu(float* out_h, float* a_h, float* b_h, int n) {
    // Initialize input vectors on device.
    float* a_d;
    float* b_d;

    cudaMalloc(&a_d, sizeof(float) * n);
    cudaMalloc(&b_d, sizeof(float) * n);

    // Initialize output vector on device.
    float* out_d;

    cudaMalloc(&out_d, sizeof(float) * n);

    // Copy input vectors from host to device memory.
    cudaMemcpy(a_d, a_h, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, sizeof(float) * n, cudaMemcpyHostToDevice);

    // Launch kernel.
    dim3 gridDim = (n + 255) / 256;
    dim3 blockDim = 256;
    vec_add_kernel<<<gridDim, blockDim>>>(out_d, a_d, b_d, n);

    // Copy output vector from device to host memory.
    cudaMemcpy(out_h, out_d, sizeof(float) * n, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(out_d);
}

int main() {
    // Initialize input vectors.
    float* a = (float*)malloc(sizeof(float) * N);
    float* b = (float*)malloc(sizeof(float) * N);

    // Initialize input values.
    for (int i = 0; i < N; i++) {
        a[i] = 3.0f;
        b[i] = 2.0f;
    }

    // Perform vector addition using CPU.
    float* cpu_output = (float*)malloc(sizeof(float) * N);
    vec_add_cpu(cpu_output, a, b, N);

    // Assert the output is correct.
    for (int i = 0; i < N; i++) {
        if (cpu_output[i] != 5.0f) {
            printf("Error: cpu_output[%d] = %f\n", i, cpu_output[i]);
            return 1;
        }
    }

    // Perform vector addition using GPU.
    float* gpu_output = (float*)malloc(sizeof(float) * N);
    vec_add_gpu(gpu_output, a, b, N);

    // Assert the output is correct.
    for (int i = 0; i < N; i++) {
        if (gpu_output[i] != 5.0f) {
            printf("Error: gpu_output[%d] = %f\n", i, gpu_output[i]);
            return 1;
        }
    }

    return 0;
}
