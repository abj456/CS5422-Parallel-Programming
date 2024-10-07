#include <cstdio>
#include <cstdlib>
// #include <mpi.h>

int main(int argc, char **argv) {
    int array_size = atoi(argv[1]);
    char *self_output_filename = argv[2];
    char *share_output_filename = argv[3];
    
    float self_out_data[array_size];
    float share_out_data[array_size];

    FILE *self_output_file = fopen(self_output_filename, "r");
    fread(self_out_data, sizeof(float), array_size, self_output_file);
    fclose(self_output_file);
    FILE *share_output_file = fopen(share_output_filename, "r");
    fread(share_out_data, sizeof(float), array_size, share_output_file);
    fclose(share_output_file);

    for(int i = 0; i < array_size; i++) {
        if(self_out_data[i] != share_out_data[i]) {
            printf("idx %d, self=%f, share=%f\n", i, self_out_data[i], share_out_data[i]);
        }
    }
    return 0;
}