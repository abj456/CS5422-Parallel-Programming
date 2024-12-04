#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <nvtx3/nvToolsExt.h>

const int INF = ((1 << 30) - 1);

//======================
#define DEV_NO 0
#define BF 64
#define HALF_BF 32
#define NUM_THREADS 32
#define STRINGIFY(x) #x
#define PRAGMA_UNROLL(x) _Pragma(STRINGIFY(unroll x))
#define addr(V, i, j) ((i) * V + (j))

cudaDeviceProp prop;
int n, m;
int org_n;
int *Dist_host;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    org_n = n;
    if(n % BF != 0) {
        n += (BF - n % BF);
    }

    Dist_host = (int*)malloc(n * n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist_host[addr(n, i, j)] = 0;
            } else {
                Dist_host[addr(n, i, j)] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist_host[addr(n, pair[0], pair[1])] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    if (n == org_n) {
        fwrite(Dist_host, sizeof(int), org_n * org_n, outfile);
    }
    else {   
        for (int i = 0; i < org_n; ++i) {
            fwrite(&Dist_host[i * n], sizeof(int), org_n, outfile);
        }
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

// __global__ void cal(
//     int n, int *Dist, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height){
//     /* common-use block_FW cal() */
//     // printf("blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, blockDim.x = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, blockDim.x);
//     int block_end_x = block_start_x + block_height;
//     int block_end_y = block_start_y + block_width;

//     int k_start = Round * B;
//     int k_end = min((Round + 1) * B, n);
    
//     for (int b_i = block_start_x + threadIdx.x; b_i < block_end_x; b_i += blockDim.x) {
//         for (int b_j = block_start_y + threadIdx.y; b_j < block_end_y; b_j += blockDim.y) {
//             // To calculate B*B elements in the block (b_i, b_j)
//             // For each block, it need to compute B times
//             for (int k = k_start; k < k_end; ++k) {
//                 // To calculate original index of elements in the block (b_i, b_j)
//                 // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
//                 int block_internal_start_x = b_i * B;
//                 int block_internal_end_x = (b_i + 1) * B;
//                 int block_internal_start_y = b_j * B;
//                 int block_internal_end_y = (b_j + 1) * B;

//                 if (block_internal_end_x > n) block_internal_end_x = n;
//                 if (block_internal_end_y > n) block_internal_end_y = n;

//                 for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
//                     for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
//                         if (Dist[i * n + k] + Dist[k * n + j] < Dist[i * n + j]) {
//                             Dist[i * n + j] = Dist[i * n + k] + Dist[k * n + j];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// __global__ void gpu_print_matrix(int V, int n, int *Dist) {
//     printf("gpu matrix\n");
//     for(int i = 0; i < V; ++i) {
//         for(int j = 0; j < V; ++j) {
//             printf("|%d %d %d|", i, j, Dist[i * n + j]);
//         }
//         printf("\n");
//     }
//     printf("gpu matrix end\n");
// }

// void print_matrix(int V, int n, int *Dist) {
//     for(int i = 0; i < V; ++i) {
//         for(int j = 0; j < V; ++j) {
//             printf("%d ", Dist[i * n + j]);
//         }
//         printf("\n");
//     }
// }

__global__ void cal_phase1(int n, int *Dist, int B, int Round) {

    int thread_i = threadIdx.y;
    int thread_j = threadIdx.x;
    int global_i = Round * BF + thread_i;
    int global_j = Round * BF + thread_j;

    __shared__ int shared_block[BF][BF];
    int global_idx0 = addr(n, global_i, global_j);
    int global_idx1 = global_idx0 + HALF_BF * n;
    int global_idx2 = global_idx0 + HALF_BF;
    int global_idx3 = global_idx0 + HALF_BF * (n + 1);
    shared_block[thread_i][thread_j] = Dist[global_idx0];
    shared_block[thread_i + HALF_BF][thread_j] = Dist[global_idx1];
    shared_block[thread_i][thread_j + HALF_BF] = Dist[global_idx2];
    shared_block[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[global_idx3];
    __syncthreads();

    /* no shared memory */
    // for(int k = Round * B; k < n && k < (Round + 1) * B; ++k){
    //     Dist[global_i * n + global_j] = min(Dist[global_i * n + global_j], (Dist[global_i * n + k] + Dist[k * n + global_j]));
    //     __syncthreads();
    // }

    PRAGMA_UNROLL(BF)
    for(int k = 0; k < BF; ++k) {
        shared_block[thread_i][thread_j] = min(shared_block[thread_i][thread_j], 
                                              (shared_block[thread_i][k] + shared_block[k][thread_j]));
        
        shared_block[thread_i + HALF_BF][thread_j] = min(shared_block[thread_i + HALF_BF][thread_j],
                                                        (shared_block[thread_i + HALF_BF][k] + shared_block[k][thread_j]));

        shared_block[thread_i][thread_j + HALF_BF] = min(shared_block[thread_i][thread_j + HALF_BF],
                                                        (shared_block[thread_i][k] + shared_block[k][thread_j + HALF_BF]));

        shared_block[thread_i + HALF_BF][thread_j + HALF_BF] = min(shared_block[thread_i + HALF_BF][thread_j + HALF_BF],
                                                                  (shared_block[thread_i + HALF_BF][k] + shared_block[k][thread_j + HALF_BF]));
        // __syncthreads();
    }
    
    Dist[global_idx0] = shared_block[thread_i][thread_j];
    Dist[global_idx1] = shared_block[thread_i + HALF_BF][thread_j];
    Dist[global_idx2] = shared_block[thread_i][thread_j + HALF_BF];
    Dist[global_idx3] = shared_block[thread_i + HALF_BF][thread_j + HALF_BF];
}

__global__ void cal_phase2(int n, int *Dist, int B, int Round) {
    if(blockIdx.x == Round) return;

    int thread_i = threadIdx.y;
    int thread_j = threadIdx.x;

    // real index in Dist
    int global_i = Round * BF + thread_i;
    int global_j = Round * BF + thread_j;
    // horizontal computation -> i fixed
    int hz_i = global_i;
    int hz_j = blockIdx.x * BF + thread_j;
    // vertical computation -> j fixed
    int vt_i = blockIdx.x * BF + thread_i;
    int vt_j = global_j;

    __shared__ int shared_pivot[BF][BF];
    __shared__ int shared_hz[BF][BF];
    __shared__ int shared_vt[BF][BF];

    int global_idx = addr(n, global_i, global_j);
    shared_pivot[thread_i][thread_j] = Dist[global_idx];
    shared_pivot[thread_i + HALF_BF][thread_j] = Dist[global_idx + HALF_BF * n];
    shared_pivot[thread_i][thread_j + HALF_BF] = Dist[global_idx + HALF_BF];
    shared_pivot[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[global_idx + HALF_BF * (n + 1)];

    int hz_idx = addr(n, hz_i, hz_j);
    shared_hz[thread_i][thread_j] = Dist[hz_idx];
    shared_hz[thread_i + HALF_BF][thread_j] = Dist[hz_idx + HALF_BF * n];
    shared_hz[thread_i][thread_j + HALF_BF] = Dist[hz_idx + HALF_BF];
    shared_hz[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[hz_idx + HALF_BF * (n + 1)];

    int vt_idx = addr(n, vt_i, vt_j);
    shared_vt[thread_i][thread_j] = Dist[vt_idx];
    shared_vt[thread_i + HALF_BF][thread_j] = Dist[vt_idx + HALF_BF * n];
    shared_vt[thread_i][thread_j + HALF_BF] = Dist[vt_idx + HALF_BF];
    shared_vt[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[vt_idx + HALF_BF * (n + 1)];
    
    __syncthreads();
    
    /* no shared memory */
    // for(int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
    //     Dist[hz_i * n + hz_j] = min(Dist[hz_i * n + hz_j], (Dist[hz_i * n + k] + Dist[k * n + hz_j]));
    //     Dist[vt_i * n + vt_j] = min(Dist[vt_i * n + vt_j], (Dist[vt_i * n + k] + Dist[k * n + vt_j]));
    //     __syncthreads();
    // }

    // #pragma unroll BF
    PRAGMA_UNROLL(BF)
    for(int k = 0; k < BF; ++k) {
        shared_hz[thread_i][thread_j] = min(shared_hz[thread_i][thread_j], 
                                            shared_pivot[thread_i][k] + shared_hz[k][thread_j]);
        shared_hz[thread_i + HALF_BF][thread_j] = min(shared_hz[thread_i + HALF_BF][thread_j], 
                                                      shared_pivot[thread_i + HALF_BF][k] + shared_hz[k][thread_j]);
        shared_hz[thread_i][thread_j + HALF_BF] = min(shared_hz[thread_i][thread_j + HALF_BF], 
                                                      shared_pivot[thread_i][k] + shared_hz[k][thread_j + HALF_BF]);
        shared_hz[thread_i + HALF_BF][thread_j + HALF_BF] = min(shared_hz[thread_i + HALF_BF][thread_j + HALF_BF], 
                                                                shared_pivot[thread_i + HALF_BF][k] + shared_hz[k][thread_j + HALF_BF]);
    
        shared_vt[thread_i][thread_j] = min(shared_vt[thread_i][thread_j],
                                            shared_vt[thread_i][k] + shared_pivot[k][thread_j]);
        shared_vt[thread_i + HALF_BF][thread_j] = min(shared_vt[thread_i + HALF_BF][thread_j],
                                                      shared_vt[thread_i + HALF_BF][k] + shared_pivot[k][thread_j]);
        shared_vt[thread_i][thread_j + HALF_BF] = min(shared_vt[thread_i][thread_j + HALF_BF],
                                                      shared_vt[thread_i][k] + shared_pivot[k][thread_j + HALF_BF]);
        shared_vt[thread_i + HALF_BF][thread_j + HALF_BF] = min(shared_vt[thread_i + HALF_BF][thread_j + HALF_BF],
                                                                shared_vt[thread_i + HALF_BF][k] + shared_pivot[k][thread_j + HALF_BF]);
        // __syncthreads();
    }
    
    Dist[hz_idx] = shared_hz[thread_i][thread_j];
    Dist[hz_idx + HALF_BF * n] = shared_hz[thread_i + HALF_BF][thread_j];
    Dist[hz_idx + HALF_BF] = shared_hz[thread_i][thread_j + HALF_BF];
    Dist[hz_idx + HALF_BF * (n + 1)] = shared_hz[thread_i + HALF_BF][thread_j + HALF_BF];

    Dist[vt_idx] = shared_vt[thread_i][thread_j];
    Dist[vt_idx + HALF_BF * n] = shared_vt[thread_i + HALF_BF][thread_j];
    Dist[vt_idx + HALF_BF] = shared_vt[thread_i][thread_j + HALF_BF];
    Dist[vt_idx + HALF_BF * (n + 1)] = shared_vt[thread_i + HALF_BF][thread_j + HALF_BF];
}

__global__ void cal_phase3(int n, int *Dist, int Round) {
    if(blockIdx.x == Round || blockIdx.y == Round) return; // skip pivot, pivot hz, pivot vt

    __shared__ int block[BF][BF];
    __shared__ int vt[BF][BF];
    __shared__ int hz[BF][BF];

    int thread_i = threadIdx.y;
    int thread_j = threadIdx.x;
    int b_i = blockIdx.y;
    int b_j = blockIdx.x;

    /* idx of block waiting to be computed */
    int block_i = b_i * BF + thread_i;
    int block_j = b_j * BF + thread_j;
    /* idx of pivot hz block */
    int hz_i = Round * BF + thread_i; // row fixed, col changed
    // int hz_j = block_j;
    /* idx of pivot vt block */
    // int vt_i = block_i;
    int vt_j = Round * BF + thread_j; // row changed, col fixed


    int block_idx = addr(n, block_i, block_j);
    block[thread_i][thread_j] = Dist[block_idx];
    block[thread_i + HALF_BF][thread_j] = Dist[block_idx + HALF_BF * n];
    block[thread_i][thread_j + HALF_BF] = Dist[block_idx + HALF_BF];
    block[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[block_idx + HALF_BF * (n + 1)];
    
    // row changed, col fixed
    int vt_idx = addr(n, block_i, vt_j);
    vt[thread_i][thread_j] = Dist[vt_idx];
    vt[thread_i + HALF_BF][thread_j] = Dist[vt_idx + HALF_BF * n];
    vt[thread_i][thread_j + HALF_BF] = Dist[vt_idx + HALF_BF];
    vt[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[vt_idx + HALF_BF * (n + 1)];

    // row fixed, col changed
    int hz_idx = addr(n, hz_i, block_j);
    hz[thread_i][thread_j] = Dist[hz_idx];
    hz[thread_i + HALF_BF][thread_j] = Dist[hz_idx + HALF_BF * n];
    hz[thread_i][thread_j + HALF_BF] = Dist[hz_idx + HALF_BF];
    hz[thread_i + HALF_BF][thread_j + HALF_BF] = Dist[hz_idx + HALF_BF * (n + 1)];

    __syncthreads();
    
    /* no shared memory */
    // for(int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
    //     Dist[global_i * n + global_j] = min(Dist[global_i * n + global_j], (Dist[global_i * n + k] + Dist[k * n + global_j]));
    //     __syncthreads();
    // }

    // #pragma unroll BF
    PRAGMA_UNROLL(BF)
    for(int k = 0; k < BF; ++k) {
        block[thread_i][thread_j] = min(block[thread_i][thread_j], 
                                        vt[thread_i][k] + hz[k][thread_j]);

        block[thread_i + HALF_BF][thread_j] = min(block[thread_i + HALF_BF][thread_j], 
                                                  vt[thread_i + HALF_BF][k] + hz[k][thread_j]);

        block[thread_i][thread_j + HALF_BF] = min(block[thread_i][thread_j + HALF_BF], 
                                                  vt[thread_i][k] + hz[k][thread_j + HALF_BF]);

        block[thread_i + HALF_BF][thread_j + HALF_BF] = min(block[thread_i + HALF_BF][thread_j + HALF_BF], 
                                                            vt[thread_i + HALF_BF][k] + hz[k][thread_j + HALF_BF]);
        // __syncthreads();
    }
    
    // Dist[addr(n, global_i, global_j)] = data0;
    // Dist[addr(n, global_i + HALF_BF, global_j)] = data1;
    // Dist[addr(n, global_i, global_j + HALF_BF)] = data2;
    // Dist[addr(n, global_i + HALF_BF, global_j + HALF_BF)] = data3;

    Dist[block_idx] = block[thread_i][thread_j];
    Dist[block_idx + HALF_BF * n] = block[thread_i + HALF_BF][thread_j];
    Dist[block_idx + HALF_BF] = block[thread_i][thread_j + HALF_BF];
    Dist[block_idx + HALF_BF * (n + 1)] = block[thread_i + HALF_BF][thread_j + HALF_BF];
}

__global__ void Phase1(int* d, int round, int V){
    // put data to shared memory
    __shared__ int shared[BF][BF];
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * BF;
    int idx_y = j + round * BF;
    int idx_d = idx_y * V + idx_x;
    shared[j][i] = d[idx_d];
    shared[j + 32][i] = d[idx_d + V * 32];
    shared[j][i + 32] = d[idx_d + 32];
    shared[j + 32][i + 32] = d[idx_d + V * 32 + 32];
    __syncthreads();
    // calculation
    #pragma unroll
    for(int k = 0; k < BF; k ++){
        shared[j][i] = min(shared[j][i], shared[j][k] + shared[k][i]);
        shared[j + 32][i] = min(shared[j + 32][i], shared[j + 32][k] + shared[k][i]);
        shared[j][i + 32] = min(shared[j][i + 32], shared[j][k] + shared[k][i + 32]);
        shared[j + 32][i + 32] = min(shared[j + 32][i + 32], shared[j + 32][k] + shared[k][i + 32]);
        __syncthreads();
    }

    //write back
    d[idx_d] = shared[j][i];
    d[idx_d + V * 32] = shared[j + 32][i];
    d[idx_d + 32] = shared[j][i + 32];
    d[idx_d + V * 32 + 32] = shared[j + 32][i + 32];
}

__global__ void Phase2(int* d, int round, int V){
    if(round == blockIdx.y)return;
    // put data to shared memory
    __shared__ int pivot[BF][BF];
    __shared__ int row[BF][BF];
    __shared__ int col[BF][BF];
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * BF;
    int idx_y = j + round * BF;
    int idx_x_row = i + blockIdx.y * BF; //y is fixed, only x change
    int idx_y_col = j + blockIdx.y * BF; //x is fixed, only y change
    // pivot
    int idx_d = idx_y * V + idx_x;
    pivot[j][i] = d[idx_d];
    pivot[j + 32][i] = d[idx_d + V * 32];
    pivot[j][i + 32] = d[idx_d + 32];
    pivot[j + 32][i + 32] = d[idx_d + V * 32 + 32];
    //row --> y fixed
    int idx_row_d = idx_y * V + idx_x_row;
    row[j][i] = d[idx_row_d];
    row[j + 32][i] = d[idx_row_d + V * 32];
    row[j][i + 32] = d[idx_row_d + 32];
    row[j + 32][i + 32] = d[idx_row_d + V * 32 + 32];
    //col --> x fixed
    int idx_col_d = idx_y_col * V + idx_x;
    col[j][i] = d[idx_col_d];
    col[j + 32][i] = d[idx_col_d + V * 32];
    col[j][i + 32] = d[idx_col_d + 32];
    col[j + 32][i + 32] = d[idx_col_d + V * 32 + 32];
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < BF; k ++){
        row[j][i] = min(row[j][i], pivot[j][k] + row[k][i]);
        row[j + 32][i] = min(row[j + 32][i], pivot[j + 32][k] + row[k][i]);
        row[j][i + 32] = min(row[j][i + 32], pivot[j][k] + row[k][i + 32]);
        row[j + 32][i + 32] = min(row[j + 32][i + 32], pivot[j + 32][k] + row[k][i + 32]);

        col[j][i] = min(col[j][i], col[j][k] + pivot[k][i]);
        col[j + 32][i] = min(col[j + 32][i], col[j + 32][k] + pivot[k][i]);
        col[j][i + 32] = min(col[j][i + 32], col[j][k] + pivot[k][i + 32]);
        col[j + 32][i + 32] = min(col[j + 32][i + 32], col[j + 32][k] + pivot[k][i + 32]);
        __syncthreads();
    }

    //row --> y fixed
    d[idx_row_d] = row[j][i];
    d[idx_row_d + V * 32] = row[j + 32][i];
    d[idx_row_d + 32] = row[j][i + 32];
    d[idx_row_d + V * 32 + 32] = row[j + 32][i + 32];
    //col --> x fixed
    d[idx_col_d] = col[j][i];
    d[idx_col_d + V * 32] = col[j + 32][i];
    d[idx_col_d + 32] = col[j][i + 32];
    d[idx_col_d + V * 32 + 32] = col[j + 32][i + 32];
}

__global__ void Phase3(int* d, int round, int V){
    // put data to shared memory
    if(round == blockIdx.x || round == blockIdx.y)return;
    __shared__ int pivot[BF][BF];
    __shared__ int row[BF][BF];
    __shared__ int col[BF][BF];
    int i = threadIdx.x;
    int j = threadIdx.y;
    // real index in d
    int idx_x = i + round * BF;
    int idx_y = j + round * BF;
    int idx_x_row = i + blockIdx.x * BF; //y is fixed, only x change
    int idx_y_col = j + blockIdx.y * BF; //x is fixed, only y change
    // pivot
    int idx_d = idx_y_col * V + idx_x_row;
    pivot[j][i] = d[idx_d];
    pivot[j + 32][i] = d[idx_d + V * 32];
    pivot[j][i + 32] = d[idx_d + 32];
    pivot[j + 32][i + 32] = d[idx_d + V * 32 + 32];
    //vt == row --> y fixed
    int idx_row_d = idx_y_col * V + idx_x; // row changed, col fixed
    row[j][i] = d[idx_row_d];
    row[j + 32][i] = d[idx_row_d + V * 32];
    row[j][i + 32] = d[idx_row_d + 32];
    row[j + 32][i + 32] = d[idx_row_d + V * 32 + 32];
    //hz == col --> x fixed
    int idx_col_d = idx_y * V + idx_x_row; // row fixed, col changed
    col[j][i] = d[idx_col_d];
    col[j + 32][i] = d[idx_col_d + V * 32];
    col[j][i + 32] = d[idx_col_d + 32];
    col[j + 32][i + 32] = d[idx_col_d + V * 32 + 32];
    __syncthreads();
    #pragma unroll
    for(int k = 0; k < BF; k ++){
        pivot[j][i] = min(pivot[j][i], row[j][k] + col[k][i]);
        pivot[j + 32][i] = min(pivot[j + 32][i], row[j + 32][k] + col[k][i]);
        pivot[j][i + 32] = min(pivot[j][i + 32], row[j][k] + col[k][i + 32]);
        pivot[j + 32][i + 32] = min(pivot[j + 32][i + 32], row[j + 32][k] + col[k][i + 32]);
    }

    d[idx_d] = pivot[j][i];
    d[idx_d + V * 32] = pivot[j + 32][i];
    d[idx_d + 32] = pivot[j][i + 32];
    d[idx_d + V * 32 + 32] = pivot[j + 32][i + 32];
}


inline void block_FW(int n, int B, int *Dist) {
    // int round = ceil(n, B);
    int round = n / BF;
    dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);
    dim3 blocksPerGrid(round, round);
    dim3 blocks(1, round);

    for(int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r, r, 1, 1);
        cal_phase1<<<1, threadsPerBlock>>>(n, Dist, BF, r);
        // Phase1<<<1, threadsPerBlock>>>(Dist, r, n);

        /* Phase 2*/
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r, 0, r, 1);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r, r + 1, round - r - 1, 1);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, 0, r, 1, r);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r + 1, r, 1, round - r - 1);
        cal_phase2<<<round, threadsPerBlock>>>(n, Dist, BF, r);
        // Phase2<<<blocks, threadsPerBlock>>>(Dist, r, n);

        /* Phase 3*/
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, 0, 0, r, r);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, 0, r + 1, round - r - 1, r);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r + 1, 0, r, round - r - 1);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r + 1, r + 1, round - r - 1, round - r - 1);
        // nvtxRangePush("Phase3 start");
        cal_phase3<<<blocksPerGrid, threadsPerBlock>>>(n, Dist, r);
        // Phase3<<<blocksPerGrid, threadsPerBlock>>>(Dist, r, n);
        // nvtxRangePop();
    }
}

int main(int argc, char* argv[]) {
    int B = BF;

    cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    // Dist_host = (int*)malloc(V * V * sizeof(int));
    input(argv[1]);
    assert(Dist_host != nullptr);

    size_t dist_size = n * n * sizeof(int);
    cudaHostRegister(Dist_host, dist_size, cudaHostRegisterDefault);

    int *Dist_dev;
    cudaMalloc(&Dist_dev, dist_size);
    cudaMemcpy(Dist_dev, Dist_host, dist_size, cudaMemcpyHostToDevice);
    assert(Dist_dev != nullptr);

    // block_FW(B);
    block_FW(n, B, Dist_dev);
    
    cudaMemcpy(Dist_host, Dist_dev, dist_size, cudaMemcpyDeviceToHost);
    output(argv[2]);
    cudaFreeHost(Dist_host);
    cudaFree(Dist_dev);
    // free(Dist_host);
    return 0;
}