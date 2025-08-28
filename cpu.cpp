#include "cpu_gemm_opt.h"
void cpu_gemm_opt(const float* A, const float* B, float* C, 
                   int M, int N, int K, float alpha, float beta) {
    #pragma omp parallel for collapse(2) // Parallelize outer loops
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = beta * C[i * N + j]; //
            for (int k = 0; k < K; k++) {
                C[i * N + j] += alpha * A[i * K + k] * B[k * N + j];
            }
        }
    }
}