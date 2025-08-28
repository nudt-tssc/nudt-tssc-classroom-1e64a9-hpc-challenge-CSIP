#include "./GPU_benchmark/gemm_opt_gpu.h"
/*
请补充具体实现,于gpu_gemm_opt中调用gpu_gemm_opt_kernel，布局自定
*/


__global__ void gpu_gemm_opt_kernel(
    float * __restrict__ a,
    float * __restrict__ b,
    float * __restrict__ c,
    const int M,
    const int N,
    const int K,
    float alpha, 
    float beta
){

}

void gpu_gemm_opt(float* d_A, float* d_B, float* d_C,
                   int M, int N, int K, float alpha, float beta){
        
}

