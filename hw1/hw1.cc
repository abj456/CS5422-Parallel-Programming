#include <cstdio>
#include <cstdlib>
#include <string.h>
#include <algorithm>
// #include <mpi.h>
#include "/opt/intel/oneapi/mpi/latest/include/mpi.h"
#include <boost/sort/spreadsort/spreadsort.hpp>
#define LEFT -1
#define RIGHT 1
#define PRINTF_ON false
#define CHECK_ARRAY false

int qsort_float_cmp(const void *a, const void *b) {
    float _a = *(float*)a;
    float _b = *(float*)b;

    if(_a < _b) return -1;
    else if (_a == _b) return 0;
    return 1;
}

inline void merge(float *local_data, float *recv_data, float *tmp,
           int local_data_size, int recv_data_size, int left_right) {
    int t_idx, l_idx, r_idx;
    if(left_right == LEFT) {
        t_idx = 0, l_idx = 0, r_idx = 0;
        while(l_idx < local_data_size && r_idx < recv_data_size){
            if(local_data[l_idx] < recv_data[r_idx]) {
                tmp[t_idx++] = local_data[l_idx++];
            }
            else {
                tmp[t_idx++] = recv_data[r_idx++];
            }
        }
        while(t_idx < local_data_size && l_idx < local_data_size) {
            tmp[t_idx++] = local_data[l_idx++];
        }
        while(t_idx < local_data_size && r_idx < recv_data_size) {
            tmp[t_idx++] = recv_data[r_idx++];
        }
    }
    else if(left_right == RIGHT) {
        l_idx = local_data_size - 1, r_idx = recv_data_size - 1, t_idx = l_idx;
        while(t_idx >= 0 && l_idx >= 0 && r_idx >= 0) {
            if(local_data[l_idx] < recv_data[r_idx]) {
                tmp[t_idx--] = recv_data[r_idx--];
            }
            else {
                tmp[t_idx--] = local_data[l_idx--];
            }
        }
        while(t_idx >= 0 && l_idx >= 0) {
            tmp[t_idx--] = local_data[l_idx--];
        }
        while(t_idx >= 0 && r_idx >= 0) {
            tmp[t_idx--] = recv_data[r_idx--];
        }
    }
    memcpy(local_data, tmp, local_data_size * sizeof(float));
    
    // int t_idx = 0, l_idx = 0, r_idx = 0;
    // while(l_idx < local_data_size && r_idx < recv_data_size){
    //     if(local_data[l_idx] < recv_data[r_idx]) {
    //         tmp[t_idx++] = local_data[l_idx++];
    //     }
    //     else {
    //         tmp[t_idx++] = recv_data[r_idx++];
    //     }
    // }
    // while(l_idx < local_data_size) {
    //     tmp[t_idx++] = local_data[l_idx++];
    // }
    // while(r_idx < recv_data_size) {
    //     tmp[t_idx++] = recv_data[r_idx++];
    // }

    // if(left_right == LEFT) {
    //     memcpy(local_data, tmp, local_data_size * sizeof(float));
    // }
    // else if(left_right == RIGHT){
    //     memcpy(local_data, tmp + recv_data_size, local_data_size * sizeof(float));
    // }
    
}

int main(int argc, char **argv) {
    if(argc != 4) {
		fprintf(stderr, "must provide exactly 3 arguments!\n");
        return 1;
    }
    MPI_Init(&argc, &argv);

    int rank, total_ranks, used_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &total_ranks);

    int array_size = atoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    float *local_data = nullptr;
    float *merged_data = nullptr;
    
    int base = array_size / total_ranks;
    int remind = array_size % total_ranks;
    used_ranks = (base == 0) ? remind : total_ranks;
    
    int *sendcounts = new int[total_ranks];
    int *displs = new int[total_ranks];
    for(int i = 0; i < total_ranks; i++) {
        sendcounts[i] = base + (i < remind ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }
    int local_data_size = sendcounts[rank];
    local_data = new float[local_data_size];
    merged_data = new float[sendcounts[0] * 2];

    
    MPI_File input_file;
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    MPI_File_read_at(input_file, sizeof(float) * displs[rank], local_data, local_data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&input_file);

    boost::sort::spreadsort::spreadsort(local_data, local_data + local_data_size);
    
    int max_recv_data_size = (rank > 0) ? sendcounts[rank - 1] : sendcounts[rank];
    float *recv_data = new float[max_recv_data_size];
    int partner, recv_data_size;
    float recv_data_first, recv_data_last;

    bool sorted = false, all_sorted = false;
    while(!all_sorted)
    { 
        all_sorted = false;
        sorted = true;
        if(local_data_size > 0) {
        // Even phase
            if(!(rank & 1) && rank < (used_ranks - 1)) { // Even ranks receive from rank+1
                partner = rank+1;
                recv_data_size = sendcounts[partner];
                // if range(rank) > range(rank + 1) -> send and merge
                MPI_Sendrecv(&local_data[local_data_size - 1], 1, MPI_FLOAT, partner, 0, 
                             &recv_data_first, 1, MPI_FLOAT, partner, 0, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if(recv_data_first < local_data[local_data_size-1]) {
                    MPI_Sendrecv(local_data, local_data_size, MPI_FLOAT, partner, 0, 
                             recv_data, recv_data_size, MPI_FLOAT, partner, 0, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(local_data, recv_data, merged_data, local_data_size, recv_data_size, LEFT);
                    sorted = false;
                }
            }
            else if (rank & 1) { // Odd ranks send to rank-1
                partner = rank-1;
                recv_data_size = sendcounts[partner];
                MPI_Sendrecv(local_data, 1, MPI_FLOAT, partner, 0,
                             &recv_data_last, 1, MPI_FLOAT, partner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if(local_data[0] < recv_data_last) {
                    MPI_Sendrecv(local_data, local_data_size, MPI_FLOAT, partner, 0,
                                recv_data, recv_data_size, MPI_FLOAT, partner, 0, 
                                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(local_data, recv_data, merged_data, local_data_size, recv_data_size, RIGHT);
                    sorted = false;
                }
            }
            // Odd phase
            if((rank & 1) && rank < (used_ranks - 1)) { // Odd ranks receive from rank+1
                partner = rank+1;
                recv_data_size = sendcounts[partner];
                MPI_Sendrecv(&local_data[local_data_size - 1], 1, MPI_FLOAT, partner, 0, 
                             &recv_data_first, 1, MPI_FLOAT, partner, 0, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                if(recv_data_first < local_data[local_data_size - 1]) {
                    MPI_Sendrecv(local_data, local_data_size, MPI_FLOAT, partner, 0,
                             recv_data, recv_data_size, MPI_FLOAT, partner, 0, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                    merge(local_data, recv_data, merged_data, local_data_size, recv_data_size, LEFT);
                    sorted = false;
                }
            }
            else if(!(rank & 1) && rank > 0) { // Even ranks send to rank-1
                partner = rank-1;
                recv_data_size = sendcounts[partner];
                MPI_Sendrecv(local_data, 1, MPI_FLOAT, partner, 0,
                             &recv_data_last, 1, MPI_FLOAT, partner, 0, 
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if(local_data[0] < recv_data_last) {
                    MPI_Sendrecv(local_data, local_data_size, MPI_FLOAT, partner, 0,
                             recv_data, recv_data_size, MPI_FLOAT, partner, 0,
                             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge(local_data, recv_data, merged_data, local_data_size, recv_data_size, RIGHT);
                    sorted = false;
                }
            }
        }
        MPI_Allreduce(&sorted, &all_sorted, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);
    }
    
    MPI_File output_file;
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * displs[rank], local_data, local_data_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);
    
    delete[] local_data;
    delete[] recv_data;
    delete[] merged_data;
    delete[] sendcounts;
    delete[] displs;
    MPI_Finalize();
    return 0;
}