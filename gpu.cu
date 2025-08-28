#include "./GPU_benchmark/gemm_opt_gpu.h"
/*
请补充具体实现,于gpu_gemm_opt中调用gpu_gemm_opt_kernel，布局自定
*/

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
    
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;


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
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[BK][BM];
    __shared__ float s_b[BK][BN];

    float r_c[TM][TN] = {0.0f};
    float r_temp_a[TM];
    float r_temp_b[TN];

    int load_a_sm_m = tid / 2;
    int load_a_sm_k = (tid % 2) * 4;
    int load_b_sm_k = tid / 32;
    int load_b_sm_n = (tid % 32) * 4;

    int load_a_gm_m = by * BM + load_a_sm_m;
    int load_b_gm_n = bx * BN + load_b_sm_n;

    for(int bk = 0; bk < (K + BK - 1) / BK; bk++){
        int load_a_gm_k = bk * BK + load_a_sm_k;
        int load_a_gm_addr = OFFSET(load_a_gm_m, load_a_gm_k, K);
        s_a[load_a_sm_k][load_a_sm_m] = a[load_a_gm_addr];
        s_a[load_a_sm_k + 1][load_a_sm_m] = a[load_a_gm_addr + 1];
        s_a[load_a_sm_k + 2][load_a_sm_m] = a[load_a_gm_addr + 2];
        s_a[load_a_sm_k + 3][load_a_sm_m] = a[load_a_gm_addr + 3];
        
        int load_b_gm_k = bk * BK + load_b_sm_k;
        int load_b_gm_addr = OFFSET(load_b_gm_k, load_b_gm_n, N);
        FLOAT4(s_b[load_b_sm_k][load_b_sm_n]) = FLOAT4(b[load_b_gm_addr]);

        __syncthreads();
        #pragma unroll
        for(int k = 0; k < BK; k++){
            FLOAT4(r_temp_a[0]) = FLOAT4(s_a[k][ty * TM / 2]);
            FLOAT4(r_temp_a[4]) = FLOAT4(s_a[k][ty * TM / 2 + BM / 2]);
            FLOAT4(r_temp_b[0]) = FLOAT4(s_b[k][tx * TN / 2]);
            FLOAT4(r_temp_b[4]) = FLOAT4(s_b[k][tx * TN / 2 + BN / 2]);
            #pragma unroll
            for(int m = 0; m < TM; m++){
                #pragma unroll
                for(int n = 0; n < TN; n++){
                    r_c[m][n] += (r_temp_a[m] * r_temp_b[n]);
                } 
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for(int i=0; i < TM / 2; i++){
        int c_gm_m = by * BM + ty * TM / 2 + i;
        int c_gm_n = bx * BN + tx * TN / 2;
        int c_gm_addr = OFFSET(c_gm_m, c_gm_n, N);
        FLOAT4(c[c_gm_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[c_gm_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }

    #pragma unroll
    for(int i=0; i < TM / 2; i++){
        int c_gm_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int c_gm_n = bx * BN + tx * TN / 2;
        int c_gm_addr = OFFSET(c_gm_m, c_gm_n, N);
        FLOAT4(c[c_gm_addr]) = FLOAT4(r_c[i + 4][0]);
        FLOAT4(c[c_gm_addr + BN / 2]) = FLOAT4(r_c[i + 4][4]);
    }
}

void gpu_gemm_opt(float* d_A, float* d_B, float* d_C,
                   int M, int N, int K, float alpha, float beta){
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / (blockSize.x * 8), 
                  (M + blockSize.y - 1) / (blockSize.y * 8));
    gpu_gemm_opt_kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
        
}

