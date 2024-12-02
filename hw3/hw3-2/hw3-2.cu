#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

const int INF = ((1 << 30) - 1);

//======================
#define DEV_NO 0
#define BF 32
cudaDeviceProp prop;
int n, m, NUM_THREADS;
int org_n;
int *Dist_host;
void print_matrix(int V, int n, int *Dist);

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
            // printf("%d %d assignment start\n", i, j);
            if (i == j) {
                Dist_host[i * n + j] = 0;
            } else {
                Dist_host[i * n + j] = INF;
            }
            // printf("%d %d assignment OK\n", i, j);
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist_host[pair[0] * n + pair[1]] = pair[2];
        // printf("n = %d, pair = (%d, %d, %d)\n", n, pair[0], pair[1], pair[2]);
    }
    // print_matrix(org_n, n, Dist_host);
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < org_n; ++i) {
        // for (int j = 0; j < org_n; ++j) {
        //     if (Dist_host[i * n + j] >= INF) Dist_host[i * n + j] = INF;
        //     // fwrite(&Dist_host[i * n + j], sizeof(int), 1, outfile);
        // }
        fwrite(&Dist_host[i * n], sizeof(int), org_n, outfile);
    }
    // fwrite(Dist_host, sizeof(int), org_n * org_n, outfile);
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

__global__ void cal(
    int n, int *Dist, int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height){
    
    // printf("blockIdx.x = %d, blockIdx.y = %d, threadIdx.x = %d, blockDim.x = %d\n", blockIdx.x, blockIdx.y, threadIdx.x, blockDim.x);
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;

    int k_start = Round * B;
    int k_end = min((Round + 1) * B, n);
    
    for (int b_i = block_start_x + threadIdx.x; b_i < block_end_x; b_i += blockDim.x) {
        for (int b_j = block_start_y + threadIdx.y; b_j < block_end_y; b_j += blockDim.y) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            for (int k = k_start; k < k_end; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                int block_internal_start_x = b_i * B;
                int block_internal_end_x = (b_i + 1) * B;
                int block_internal_start_y = b_j * B;
                int block_internal_end_y = (b_j + 1) * B;

                if (block_internal_end_x > n) block_internal_end_x = n;
                if (block_internal_end_y > n) block_internal_end_y = n;

                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (Dist[i * n + k] + Dist[k * n + j] < Dist[i * n + j]) {
                            Dist[i * n + j] = Dist[i * n + k] + Dist[k * n + j];
                        }
                    }
                }
            }
        }
    }
}

__global__ void gpu_print_matrix(int V, int n, int *Dist) {
    printf("gpu matrix\n");
    for(int i = 0; i < V; ++i) {
        for(int j = 0; j < V; ++j) {
            printf("|%d %d %d|", i, j, Dist[i * n + j]);
        }
        printf("\n");
    }
    printf("gpu matrix end\n");
}

void print_matrix(int V, int n, int *Dist) {
    for(int i = 0; i < V; ++i) {
        for(int j = 0; j < V; ++j) {
            printf("%d ", Dist[i * n + j]);
        }
        printf("\n");
    }
}

__global__ void cal_phase1(int n, int *Dist, int B, int Round) {

    int thread_i = threadIdx.x;
    int thread_j = threadIdx.y;
    int global_i = Round * BF + thread_i;
    int global_j = Round * BF + thread_j;

    // if(global_i >= n || global_j >= n) return;

    __shared__ int shared_block[BF][BF];
    shared_block[thread_i][thread_j] = Dist[global_i * n + global_j];
    __syncthreads();

    // for(int k = Round * B; k < n && k < (Round + 1) * B; ++k){
    #pragma unroll 32
    for(int k = 0; k < BF && (Round * BF + k) < n; ++k) {
        // Dist[global_i * n + global_j] = min(Dist[global_i * n + global_j], (Dist[global_i * n + k] + Dist[k * n + global_j]));
        shared_block[thread_i][thread_j] = min(shared_block[thread_i][thread_j], 
                                              (shared_block[thread_i][k] + shared_block[k][thread_j]));
        __syncthreads();
    }
    Dist[global_i * n + global_j] = shared_block[thread_i][thread_j];
}

__global__ void cal_phase2(int n, int *Dist, int B, int Round) {
    int thread_i = threadIdx.x;
    int thread_j = threadIdx.y;
    int b_i = blockIdx.x;

    if(b_i == Round) return;

    // real index in Dist
    int global_i = Round * BF + thread_i;
    int global_j = Round * BF + thread_j;
    // horizontal computation -> i fixed
    int hz_i = global_i;
    int hz_j = b_i * BF + thread_j;
    // vertical computation -> j fixed
    int vt_i = b_i * BF + thread_i;
    int vt_j = global_j;

    __shared__ int shared_pivot[BF][BF];
    __shared__ int shared_hz[BF][BF];
    __shared__ int shared_vt[BF][BF];

    shared_pivot[thread_i][thread_j] = Dist[global_i * n + global_j];
    shared_hz[thread_i][thread_j] = Dist[hz_i * n + hz_j];
    shared_vt[thread_i][thread_j] = Dist[vt_i * n + vt_j];
    
    __syncthreads();
    
    // for(int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
    //     // Dist[hz_i * n + hz_j] = min(Dist[hz_i * n + hz_j], (Dist[hz_i * n + k] + Dist[k * n + hz_j]));
    //     shared_hz[thread_i][thread_j] = min(shared_hz[thread_i][thread_j], (Dist[hz_i * n + k] + Dist[k * n + hz_j]));
    //     // Dist[vt_i * n + vt_j] = min(Dist[vt_i * n + vt_j], (Dist[vt_i * n + k] + Dist[k * n + vt_j]));
    //     shared_vt[thread_i][thread_j] = min(shared_vt[thread_i][thread_j], (Dist[vt_i * n + k] + Dist[k * n + vt_j]));
    //     __syncthreads();
    // }

    for(int k = 0; k < BF; ++k) {
        shared_hz[thread_i][thread_j] = min(shared_hz[thread_i][thread_j], 
                                            shared_pivot[thread_i][k] + shared_hz[k][thread_j]);
    
        shared_vt[thread_i][thread_j] = min(shared_vt[thread_i][thread_j],
                                            shared_vt[thread_i][k] + shared_pivot[k][thread_j]);
        __syncthreads();
    }
    
    Dist[hz_i * n + hz_j] = shared_hz[thread_i][thread_j];
    Dist[vt_i * n + vt_j] = shared_vt[thread_i][thread_j];
}

__global__ void cal_phase3(int n, int *Dist, int B, int Round) {
    int thread_i = threadIdx.x;
    int thread_j = threadIdx.y;
    int b_i = blockIdx.x;
    int b_j = blockIdx.y;

    if(b_i == Round || b_j == Round) return;

    int global_i = b_i * BF + thread_i;
    int global_j = b_j * BF + thread_j;

    __shared__ int shared_hz[BF][BF];
    __shared__ int shared_vt[BF][BF];

    
    for(int k = Round * B; k < (Round + 1) * B && k < n; ++k) {
        Dist[global_i * n + global_j] = min(Dist[global_i * n + global_j], (Dist[global_i * n + k] + Dist[k * n + global_j]));

        __syncthreads();
    }
}

void block_FW(int n, int B, int *Dist) {
    int round = ceil(n, B);
    dim3 threadsPerBlock(BF, BF);
    dim3 blocksPerGrid(round, round);
    size_t shared_block_size = BF * BF * sizeof(int);

    for(int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r, r, 1, 1);
        cal_phase1<<<1, threadsPerBlock>>>(n, Dist, B, r);
        // cudaDeviceSynchronize();

        /* Phase 2*/
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r, 0, r, 1);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r, r + 1, round - r - 1, 1);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, 0, r, 1, r);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r + 1, r, 1, round - r - 1);
        cal_phase2<<<round, threadsPerBlock>>>(n, Dist, BF, r);

        /* Phase 3*/
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, 0, 0, r, r);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, 0, r + 1, round - r - 1, r);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r + 1, 0, r, round - r - 1);
        // cal<<<1, threadsPerBlock>>>(n, Dist, B, r, r + 1, r + 1, round - r - 1, round - r - 1);
        cal_phase3<<<blocksPerGrid, threadsPerBlock>>>(n, Dist, BF, r);
    }
}

int main(int argc, char* argv[]) {
    int B = BF;

    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    // Dist_host = (int*)malloc(V * V * sizeof(int));
    input(argv[1]);
    assert(Dist_host != nullptr);
    // print_matrix(org_n, Dist_host);

    int *Dist_dev;
    cudaMalloc(&Dist_dev, n * n * sizeof(int));
    cudaMemcpy(Dist_dev, Dist_host, n * n * sizeof(int), cudaMemcpyHostToDevice);
    assert(Dist_dev != nullptr);
    // gpu_print_matrix<<<1, 1>>>(org_n, n, Dist_dev);

    // block_FW(B);
    block_FW(n, B, Dist_dev);
    
    cudaMemcpy(Dist_host, Dist_dev, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    // print_matrix(org_n, n, Dist_host);
    output(argv[2]);
    cudaFreeHost(Dist_host);
    cudaFree(Dist_dev);
    return 0;
}