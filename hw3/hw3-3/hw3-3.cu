#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>

const int INF = ((1 << 30) - 1);
double getTimeStamp() {
    struct timeval tv;
    gettimeofday( &tv, NULL );
    return (double) tv.tv_usec/1000000 + tv.tv_sec;
}

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
int ncpus;
size_t dist_size;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    
    org_n = n;
    if(n % BF != 0) {
        n += (BF - n % BF);
    }
    printf("# of vertex: %d\n", org_n);

    dist_size = n * n * sizeof(int);
    Dist_host = (int*)malloc(dist_size);

    #pragma omp parallel for collapse(2)
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

__global__ void cal_phase3(int n, int *Dist, int Round, int row_offset) {
    if(blockIdx.x == Round || blockIdx.y + row_offset == Round) return; // skip pivot, pivot hz, pivot vt

    __shared__ int block[BF][BF];
    __shared__ int vt[BF][BF];
    __shared__ int hz[BF][BF];

    int thread_i = threadIdx.y;
    int thread_j = threadIdx.x;
    int b_i = blockIdx.y + row_offset;
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

    Dist[block_idx] = block[thread_i][thread_j];
    Dist[block_idx + HALF_BF * n] = block[thread_i + HALF_BF][thread_j];
    Dist[block_idx + HALF_BF] = block[thread_i][thread_j + HALF_BF];
    Dist[block_idx + HALF_BF * (n + 1)] = block[thread_i + HALF_BF][thread_j + HALF_BF];
}

inline void block_FW(int n, int B, int **Dist) {
    
    // int round = ceil(n, B);
    int round = n / BF;
    dim3 threadsPerBlock(NUM_THREADS, NUM_THREADS);
    dim3 blocks(1, round);
    // dim3 blocksPerGrid(round, round);

    #pragma omp parallel num_threads(2)
    {
        unsigned int cpu_tid = omp_get_thread_num();
        unsigned int half_round = round / 2;
        unsigned int row_offset = (cpu_tid) ? half_round : 0;
        unsigned int div_row = (cpu_tid) ? (round - half_round) : half_round;
        
        cudaSetDevice(cpu_tid);
        cudaMalloc(&Dist[cpu_tid], dist_size);
        assert(Dist[cpu_tid] != nullptr);
        cudaMemcpy(Dist[cpu_tid], Dist_host, dist_size, cudaMemcpyHostToDevice);
        dim3 blocksPerGrid(round, div_row);
        
        for(int r = 0; r < round; ++r) {
            if(r >= row_offset && r < row_offset + div_row) {
                cudaMemcpy(Dist[!cpu_tid] + r * BF * n, Dist[cpu_tid] + r * BF * n, BF * n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            #pragma omp barrier

            /* Phase 1*/
            cal_phase1<<<1, threadsPerBlock>>>(n, Dist[cpu_tid], BF, r);
            /* Phase 2*/
            cal_phase2<<<round, threadsPerBlock>>>(n, Dist[cpu_tid], BF, r);
            /* Phase 3*/
            cal_phase3<<<blocksPerGrid, threadsPerBlock>>>(n, Dist[cpu_tid], r, row_offset);
        }
        cudaMemcpy(Dist_host + row_offset * BF * n, Dist[cpu_tid] + (row_offset * BF * n), 
                   div_row * BF * n * sizeof(int), cudaMemcpyDeviceToHost);
        #pragma omp barrier
        
        cudaFree(Dist[cpu_tid]);
    }
}

int main(int argc, char* argv[]) {
    int B = BF;
    /* detect how many CPUs are available */
    // cpu_set_t cpu_set;
    // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    // printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    // int ncpus = CPU_COUNT(&cpu_set);

    cudaGetDeviceProperties(&prop, DEV_NO);
    // printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);

    // double total_IO_time = 0.0;

    // double before = getTimeStamp();
    input(argv[1]);
    // double after = getTimeStamp();
    // total_IO_time += (after - before);
    assert(Dist_host != nullptr);

    cudaHostRegister(Dist_host, dist_size, cudaHostRegisterDefault);

    int *Dist_dev[2];

    // block_FW(B);
    block_FW(n, B, Dist_dev);
    
    // cudaMemcpy(Dist_host, Dist_dev, dist_size, cudaMemcpyDeviceToHost);
    // before = getTimeStamp();
    output(argv[2]);
    // after = getTimeStamp();
    // total_IO_time += (after - before);
    // printf("total IO time: %lf\n", total_IO_time);

    cudaFreeHost(Dist_host);
    return 0;
}