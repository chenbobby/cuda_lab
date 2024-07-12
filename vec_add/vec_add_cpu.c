#include <bits/time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N (10 * 1000 * 1000)

void vec_add_cpu(float* out, float* a, float* b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
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

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    vec_add_cpu(cpu_output, a, b, N);
    clock_gettime(CLOCK_MONOTONIC, &end);
    long delta =
        (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec);

    printf("vec_add_cpu time: %.10lins\n", delta);

    // Assert the output is correct.
    for (int i = 0; i < N; i++) {
        if (cpu_output[i] != 5.0f) {
            printf("Error: cpu_output[%d] = %f\n", i, cpu_output[i]);
            return 1;
        }
    }

    return 0;
}
