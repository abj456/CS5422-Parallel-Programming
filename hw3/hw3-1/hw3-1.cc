#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <omp.h>
#include <algorithm>
#define CHUNK_SIZE 10

const int INF = ((1 << 30) - 1);
const int V = 50010;
void input(char* inFileName);
void output(char* outFileName);

void Floyd_Warshall();
void block_FW(int B);
int ceil(int a, int b);
void cal(int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height);

int n, m, NUM_THREADS;
static int Dist[V][V];

int main(int argc, char* argv[]) {
    input(argv[1]);
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    NUM_THREADS = CPU_COUNT(&cpu_set);
    omp_set_num_threads(NUM_THREADS);
    int B = 512;
    block_FW(B);
    // Floyd_Warshall();
    output(argv[2]);
    return 0;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    printf("n: %d, m: %d\n", n, m);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    #pragma unroll 5
    for (int i = 0; i < n; ++i) {
        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
        for (int j = 0; j < n; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void Floyd_Warshall(){
    for(int k = 0; k < n; ++k) {
        #pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
        for(int row = 0; row < n; ++row) {
            for(int col = 0; col < n; ++col) {
                Dist[row][col] = std::min(Dist[row][col], Dist[row][k] + Dist[k][col]);
            }
        }
    }
}

void block_FW(int B) {
    int round = ceil(n, B);

    for (int r = 0; r < round; ++r) {
        printf("%d %d\n", r, round);
        fflush(stdout);
        /* Phase 1*/
        cal(B, r, r, r, 1, 1);

        /* Phase 2*/
        #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
        for(int i = 0; i < round; ++i) {
            if(i != r) {
                cal(B, r, r, i, 1, 1);
                cal(B, r, i, r, 1, 1);
            }
        }
        // cal(B, r, r, 0, r, 1);
        // cal(B, r, r, r + 1, round - r - 1, 1);
        // cal(B, r, 0, r, 1, r);
        // cal(B, r, r + 1, r, 1, round - r - 1);

        /* Phase 3*/
        #pragma omp parallel for num_threads(NUM_THREADS) collapse(2) schedule(dynamic)
        for(int i = 0; i < round; ++i){
            for(int j = 0; j < round; ++j) {
                if(i != r && j != r) {
                    cal(B, r, i, j, 1, 1);
                }
            }
        }
        // cal(B, r, 0, 0, r, r);
        // cal(B, r, 0, r + 1, round - r - 1, r);
        // cal(B, r, r + 1, 0, r, round - r - 1);
        // cal(B, r, r + 1, r + 1, round - r - 1, round - r - 1);
    }
}

void cal(
    int B, int Round, int block_start_x, int block_start_y, int block_width, int block_height) {
    int block_end_x = block_start_x + block_height;
    int block_end_y = block_start_y + block_width;
    int k_start = Round * B, k_end = std::min((Round + 1) * B, n);

    
    for (int b_i = block_start_x; b_i < block_end_x; ++b_i) {
        for (int b_j = block_start_y; b_j < block_end_y; ++b_j) {
            // To calculate B*B elements in the block (b_i, b_j)
            // For each block, it need to compute B times
            int block_internal_start_x = b_i * B;
            int block_internal_end_x = std::min((b_i + 1) * B, n);
            int block_internal_start_y = b_j * B;
            int block_internal_end_y = std::min((b_j + 1) * B, n);

            // if (block_internal_end_x > n) block_internal_end_x = n;
            // if (block_internal_end_y > n) block_internal_end_y = n;

            for (int k = k_start; k < k_end; ++k) {
                // To calculate original index of elements in the block (b_i, b_j)
                // For instance, original index of (0,0) in block (1,2) is (2,5) for V=6,B=2
                for (int i = block_internal_start_x; i < block_internal_end_x; ++i) {
                    for (int j = block_internal_start_y; j < block_internal_end_y; ++j) {
                        if (Dist[i][k] + Dist[k][j] < Dist[i][j]) {
                            Dist[i][j] = Dist[i][k] + Dist[k][j];
                        }
                    }
                }
            }
        }
    }
}
