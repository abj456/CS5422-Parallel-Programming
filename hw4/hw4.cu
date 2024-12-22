#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <cuda.h>
// #include <nvtx3/nvToolsExt.h>

#define DEV_NO 0
#define BlockFactor 32
#define BR BlockFactor
#define BC BlockFactor

void input(char *input_filename);
void output(char *output_filename);
void flash_attention(float *q, float *k, float *v, float *o);
__global__ void gpu_flash_attn(float *l, float *m, int N, int d, int j,
                               float *q, float *k, float *v, float *o
                            //    float *cuda_sij, float *cuda_mij, float *cuda_pij, float *cuda_lij, 
                            //    float *cuda_kj, float *cuda_vj
                               );

void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar);
void RowMax(float *out, float *in, int br, int bc);
void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
void RowSum(float *out, float *in, int br, int bc);
void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc);

// __device__ void gpu_QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar, int d);
// __device__ void gpu_RowMax(float *out, float *in, int br, int bc);
// __device__ void gpu_MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc);
// __device__ void gpu_RowSum(float *out, float *in, int br, int bc);
// __device__ void gpu_UpdateMiLiOi(float *mi, float *li, float *oi, float *oi_tmp, float *mij, float *lij, float *pij, float *vj, int br, int bc, int d);

/* DEBUG FUNC */
__global__ void gpu_checkQKVO(float *in, int B, int N, int d);
void checkCPU_GPU(float *O_cpu, float *O_gpu, int _B, int _N, int _d, char *matrix_name);

float _max(float a, float b) { return a > b ? a : b; }
float _min(float a, float b) { return a < b ? a : b; }
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

int B, N, d;
float *Q, *K, *V, *O;
float *Q_gpu, *K_gpu, *V_gpu, *O_gpu;
float *l, *m;
// float *O_tmp;

/* DEBUG ARRAY */
// float *cpu_sij, *cpu_mij, *cpu_pij, *cpu_lij;
// float *gpu_sij, *gpu_mij, *gpu_pij, *gpu_lij;
// float *cuda_sij, *cuda_mij, *cuda_pij, *cuda_lij;
// float *gpu_kj, *gpu_vj;
// float *cuda_kj, *cuda_vj;

void print_rc(int rc) {
    printf("%d\n", rc);
    return;
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <input_filename> <output_filename>\n", argv[0]);
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("testcase: %s\n", argv[1]);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %lu\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    // printf("maxBlocksPerMultiProcessor = %d, totalGlobalMem = %lu\n", prop.maxBlocksPerMultiProcessor, prop.totalGlobalMem);


    input(argv[1]);
    size_t QKVO_size = B * N * d * sizeof(float);
    // printf("size of malloc = %lu\n", sizeof(float) * (BlockFactor * d * 4 + BlockFactor * 4 + BlockFactor * BlockFactor * 2));
    // printf("size of GlobalMem = %lu\n", QKVO_size * 4);
    // printf("(B, N, d): (%d, %d, %d)\n", B, N, d);

    cudaMalloc(&Q_gpu, QKVO_size);
    cudaMalloc(&K_gpu, QKVO_size);
    cudaMalloc(&V_gpu, QKVO_size);
    cudaMalloc(&O_gpu, QKVO_size);
    cudaMemcpy(Q_gpu, Q, QKVO_size, cudaMemcpyHostToDevice);
    cudaMemcpy(K_gpu, K, QKVO_size, cudaMemcpyHostToDevice);
    cudaMemcpy(V_gpu, V, QKVO_size, cudaMemcpyHostToDevice);
    cudaMemcpy(O_gpu, O, QKVO_size, cudaMemcpyHostToDevice);

    cudaMalloc(&l, N * sizeof(float));
    cudaMalloc(&m, N * sizeof(float));

    int br = BR, bc = BC;
    int tr = N / br, tc = N / bc;
    dim3 blocksPerGrid(1, tr);
    dim3 threadsPerBlock(bc, br);
    size_t sizeof_kj_vj_qi_oi = (bc * d * 2 + br * d * 2) * sizeof(float);
    // printf("size of kj_vj_qi_oi_oitmp = %lu\n", sizeof_kj_vj_qi_oi);
    // printf("size of static shared mem = %lu\n", sizeof(float) * (BR * 6 + BR * BC * 2));

    double start, end;
    start = getTimeStamp();

    for (int i = 0; i < B; i++) {
        cudaMemset(l, 0x00, N * sizeof(float));
        cudaMemset(m, FLT_MIN, N * sizeof(float));
        
        for(int j = 0; j < tc; ++j)
            gpu_flash_attn<<<blocksPerGrid, threadsPerBlock, sizeof_kj_vj_qi_oi>>>(
                l, m, N, d, j,
                Q_gpu + (i * N * d), 
                K_gpu + (i * N * d), 
                V_gpu + (i * N * d), 
                O_gpu + (i * N * d)
            );
        
        // flash_attention(
        //     Q + (i * N * d), 
        //     K + (i * N * d), 
        //     V + (i * N * d), 
        //     O + (i * N * d)
        // );
    }

    end = getTimeStamp();
    printf("Time: %.3f seconds\n", end - start);

    cudaMemcpy(O, O_gpu, B * N * d * sizeof(float), cudaMemcpyDeviceToHost);
    
    // cudaError_t err = cudaGetLastError(); 
    // if ( err != cudaSuccess ) { 
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    //     return 0;
    // }

    // checkCPU_GPU(O, O_tmp, B, N, d, "O");

    output(argv[2]);

    return 0;
}

void input(char *input_filename) {
    FILE *file = fopen(input_filename, "rb");

    fread(&B, sizeof(int), 1, file);
    fread(&N, sizeof(int), 1, file);
    fread(&d, sizeof(int), 1, file);

    Q = (float *)malloc(B * N * d * sizeof(float));
    K = (float *)malloc(B * N * d * sizeof(float));
    V = (float *)malloc(B * N * d * sizeof(float));
    O = (float *)malloc(B * N * d * sizeof(float));
    // O_tmp = (float *)malloc(B * N * d * sizeof(float));

    for (int i = 0; i < B; i++) {
        fread(Q + (i * N * d), sizeof(float), N * d, file);
        fread(K + (i * N * d), sizeof(float), N * d, file);
        fread(V + (i * N * d), sizeof(float), N * d, file);
    }
    memset(O, 0x00, B * N * d * sizeof(float));
    // memset(O_tmp, 0x00, B * N * d * sizeof(float));

    fclose(file);
}

void output(char *output_filename) {
    FILE *file = fopen(output_filename, "wb");

    fwrite(O, sizeof(float), B * N * d, file);

    free(Q);
    free(K);
    free(V);
    free(O);
    cudaFree(Q_gpu);
    cudaFree(K_gpu);
    cudaFree(V_gpu);
    cudaFree(O_gpu);
    cudaFree(l);
    cudaFree(m);
    // free(O_tmp);

    fclose(file);
}

void checkCPU_GPU(float *O_cpu, float *O_gpu, int _B, int _N, int _d, char *matrix_name) {
    for(int i = 0; i < _B; ++i) {
        for(int j = 0; j < _N; ++j) {
            for(int k = 0; k < _d; ++k) {
                long long idx = i * N * d + j * d + k; 
                if(O_cpu[idx] != O_gpu[idx]) {
                    printf("cpu_%s[%d][%d][%d] diff, cpu = %f, gpu = %f\n", matrix_name, i, j, k, O_cpu[idx], O_gpu[idx]);
                    // return;
                }
            }
        }
    }
    return;
}

__global__ void gpu_flash_attn(float *l, float *m, int N, int d, int j,
                               float *q, float *k, float *v, float *o
                               ) {
    extern __shared__ float s_mem[];
    float *kj = s_mem;
    float *vj = kj + BC * d;
    float *qi = vj + BC * d;
    float *oi = qi + BR * d;

    __shared__ float li[BR];
    __shared__ float mi[BR];
    __shared__ float sij[BR * BC];
    __shared__ float pij[BR * BC];
    __shared__ float mij[BR];
    __shared__ float lij[BR];
    
    int thd_i = threadIdx.y;
    int thd_j = threadIdx.x;

    // row = blockIdx.y * BR + thd_i;
    int row = blockIdx.y * BR + thd_i;
    
    int d_stride = d / BR;
    // for(int idx = 0; idx < d; ++idx) { // only parallelize N
    //     kj[thd_j * d + idx] = k[j * BC * d + thd_j * d + idx];
    //     vj[thd_j * d + idx] = v[j * BC * d + thd_j * d + idx];
    // }
    #pragma unroll
    for(int idx = 0; idx < d_stride; ++idx) {    
        kj[thd_j * d + idx * BR + thd_i] = k[j * BC * d + thd_j * d + idx * BR + thd_i];
        vj[thd_j * d + idx * BR + thd_i] = v[j * BC * d + thd_j * d + idx * BR + thd_i];
    }

    d_stride = d / BC;
    #pragma unroll
    for(int idx = 0; idx < d_stride; ++idx) {
        qi[thd_i * d + idx * BC + thd_j] = q[row * d + idx * BC + thd_j];
        oi[thd_i * d + idx * BC + thd_j] = o[row * d + idx * BC + thd_j];
    }
    li[thd_i] = l[row];
    mi[thd_i] = m[row];
    __syncthreads();

    /* gpu_QKDotAndScalar(sij, qi, kj, BR, BC, 1.0 / sqrt(double(d)), d) */
    sij[thd_i * BC + thd_j] = 0.0F;
    for(int idx = 0; idx < d; ++idx) {
        sij[thd_i * BC + thd_j] += qi[thd_i * d + idx] * kj[thd_j * d + idx];
    }
    float scalar = 1.0 / sqrt(double(d));
    sij[thd_i * BC + thd_j] *= scalar;
    // __syncthreads();

    /* gpu_RowMax(mij, sij, BR, BC) */
    mij[thd_i] = sij[thd_i * BC];
    for(int idx = 0; idx < BC; ++idx) {
        mij[thd_i] = max(mij[thd_i], sij[thd_i * BC + idx]);
    }
    // __syncthreads();
    
    /* gpu_MinusMaxAndExp(pij, sij, mij, BR, BC) */
    pij[thd_i * BC + thd_j] = exp(sij[thd_i * BC + thd_j] - mij[thd_i]);
    // __syncthreads();

    /* gpu_RowSum(lij, pij, BR, BC) */
    lij[thd_i] = 0.0F;
    for (int idx = 0; idx < BC; idx++) {
        lij[thd_i] += pij[thd_i * BC + idx];
    }
    // __syncthreads();

    /* gpu_UpdateMiLiOi(mi, li, oi, oi_tmp, mij, lij, pij, vj, BR, BC, d) */ 
    __shared__ float mi_new[BR];
    __shared__ float li_new[BR];

    mi_new[thd_i] = max(mi[thd_i], mij[thd_i]);
    float coeff_old = exp(mi[thd_i] - mi_new[thd_i]);
    float coeff_cur = exp(mij[thd_i] - mi_new[thd_i]);
    li_new[thd_i] = coeff_old * li[thd_i] + coeff_cur * lij[thd_i];
    // __syncthreads();

    d_stride = d / BC;
    #pragma unroll
    for(int idx = 0; idx < d_stride; ++idx) {
        float pv = 0.0F;
        for(int t = 0; t < BC; ++t) {
            pv += pij[thd_i * BC + t] * vj[t * d + idx * BC + thd_j];
        }
        oi[thd_i * d + idx * BC + thd_j] = (li[thd_i] * coeff_old * oi[thd_i * d + idx * BC + thd_j] + coeff_cur * pv) / li_new[thd_i];
    }
    __syncthreads();
    
    #pragma unroll
    for(int idx = 0; idx < d_stride; ++idx) {
        o[row * d + idx * BC + thd_j] = oi[thd_i * d + idx * BC + thd_j];
    }
    l[row] = li_new[thd_i];
    m[row] = mi_new[thd_i];
    // __syncthreads();

    // if(j == 1 && blockIdx.y == 0 && thd_i == 0 && thd_j == 0) {
    //     for(int s = 0; s < BR; ++s) {
    //         for(int t = 0; t < d; ++t) {
    //             // printf("GPU %s[%d][%d] = %f\n", "kj", s, t, kj[s * d + t]);
    //             // printf("GPU %s[%d][%d] = %f\n", "vj", s, t, vj[s * d + t]);
    //             // printf("GPU %s[%d][%d] = %f\n", "qi", s, t, qi[s * d + t]);
    //             // printf("GPU %s[%d][%d] = %f\n", "sij", s, t, sij[s * d + t]);
    //             printf("GPU %s[%d][%d] = %f\n", "oi", s, t, oi[s * d + t]);
    //         }
    //         // printf("GPU %s[%d] = %f\n", "li", s, li_new[s]);
    //     }
    // }
    // __syncthreads();
}

void flash_attention(float *q, float *k, float *v, float *o) {
    float *l = (float *)malloc(N * sizeof(float));
    float *m = (float *)malloc(N * sizeof(float));
    memset(l, 0x00, N * sizeof(float));
    memset(m, FLT_MIN, N * sizeof(float));
    // for (int i = 0; i < N; i++) {
    //     m[i] = FLT_MIN;
    // }

    int br = BlockFactor, bc = BlockFactor;
    int tr = N / br, tc = N / bc;
    float *kj = (float *)malloc(bc * d * sizeof(float));
    float *vj = (float *)malloc(bc * d * sizeof(float));
    float *qi = (float *)malloc(br * d * sizeof(float));
    float *oi = (float *)malloc(br * d * sizeof(float));
    float *li = (float *)malloc(br * sizeof(float));
    float *mi = (float *)malloc(br * sizeof(float));

    float *sij = (float *)malloc(br * bc * sizeof(float));
    float *pij = (float *)malloc(br * bc * sizeof(float));
    float *mij = (float *)malloc(br * sizeof(float));
    float *lij = (float *)malloc(br * sizeof(float));

    for (int j = 0; j < tc; j++) {
        memcpy(kj, k + j * bc * d, bc * d * sizeof(float));
        memcpy(vj, v + j * bc * d, bc * d * sizeof(float));

        for (int i = 0; i < tr; i++) {
            memcpy(qi, q + i * br * d, br * d * sizeof(float));
            memcpy(oi, o + i * br * d, br * d * sizeof(float));
            memcpy(li, l + i * br, br * sizeof(float));
            memcpy(mi, m + i * br, br * sizeof(float));

            QKDotAndScalar(sij, qi, kj, br, bc, 1.0 / sqrt(d));
            RowMax(mij, sij, br, bc);
            MinusMaxAndExp(pij, sij, mij, br, bc);
            RowSum(lij, pij, br, bc);

            UpdateMiLiOi(mi, li, oi, mij, lij, pij, vj, br, bc);

            memcpy(o + i * br * d, oi, br * d * sizeof(float));
            memcpy(l + i * br, li, br * sizeof(float));
            memcpy(m + i * br, mi, br * sizeof(float));

            // if(j == 1 && i == 0) {
            //     for(int s = 0; s < br; ++s) {
            //         for(int t = 0; t < d; ++t) {
            //             // printf("CPU %s[%d][%d] = %f\n", "kj", s, t, kj[s * d + t]);
            //             // printf("CPU %s[%d][%d] = %f\n", "vj", s, t, vj[s * d + t]);
            //             // printf("CPU %s[%d][%d] = %f\n", "qi", s, t, qi[s * d + t]);
            //             // printf("CPU %s[%d][%d] = %f\n", "sij", s, t, sij[s * d + t]);
            //             printf("CPU %s[%d][%d] = %f\n", "oi", s, t, oi[s * d + t]);
            //         }
            //         // printf("CPU %s[%d] = %f\n", "li", s, li[s]);
            //     }
            // }
        }
            
    }

    // memcpy(cpu_sij, sij, br * bc * sizeof(float));
    // memcpy(cpu_mij, mij, br * sizeof(float));
    // memcpy(cpu_pij, pij, br * bc * sizeof(float));
    // memcpy(cpu_lij, lij, br * sizeof(float));

    free(sij);
    free(pij);
    free(mij);
    free(lij);

    free(kj);
    free(vj);
    free(qi);
    free(oi);
    free(li);
    free(mi);

    free(l);
    free(m);
}

void QKDotAndScalar(float *out, float *q, float *k, int br, int bc, float scalar) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = 0.0F;
            for (int t = 0; t < d; t++) {
                out[i * bc + j] += q[i * d + t] * k[j * d + t];
            }
            out[i * bc + j] *= scalar;
        }
    }
}

void RowMax(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = in[i * bc];
        for (int j = 0; j < bc; j++) {
            out[i] = _max(out[i], in[i * bc + j]);
        }
    }
}

void MinusMaxAndExp(float *out, float *in, float *mx, int br, int bc) {
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            out[i * bc + j] = exp(in[i * bc + j] - mx[i]);
        }
    }
}

void RowSum(float *out, float *in, int br, int bc) {
    for (int i = 0; i < br; i++) {
        out[i] = 0.0F;
        for (int j = 0; j < bc; j++) {
            out[i] += in[i * bc + j];
        }
    }
}

void UpdateMiLiOi(float *mi, float *li, float *oi, float *mij, float *lij, float *pij, float *vj, int br, int bc) {
    float *mi_new = (float *)malloc(br * sizeof(float));
    float *li_new = (float *)malloc(br * sizeof(float));

    for (int i = 0; i < br; i++) {
        mi_new[i] = _max(mi[i], mij[i]);
        li_new[i] = exp(mi[i] - mi_new[i]) * li[i] + exp(mij[i] - mi_new[i]) * lij[i];
    }

    for (int i = 0; i < br; i++) {
        for (int j = 0; j < d; j++) {
            float pv = 0.0F;
            for (int t = 0; t < bc; t++) {
                pv += pij[i * bc + t] * vj[t * d + j];
            }
            oi[i * d + j] = (li[i] * exp(mi[i] - mi_new[i]) * oi[i * d + j] + exp(mij[i] - mi_new[i]) * pv) / li_new[i];
        }
    }

    memcpy(mi, mi_new, br * sizeof(float));
    memcpy(li, li_new, br * sizeof(float));
    
    free(mi_new);
    free(li_new);
}

