#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

void gpu_gemm_opt(float* d_A, float* d_B, float* d_C,
                   int M, int N, int K, float alpha, float beta);


