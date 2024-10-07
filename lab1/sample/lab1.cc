#include "mpi.h"
#include <assert.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long r_sqr = r * r;
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	unsigned long long global_pixels = 0;

	int i, rank, size;
	MPI_Status stat;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	unsigned long long x_start = rank * (r / size);
	unsigned long long x_end = (rank < (size - 1)) ? (rank + 1) * (r / size) : r;

	for(unsigned long long x = x_start; x < x_end; x++) {
		unsigned long long tmp = r_sqr - x*x;
		unsigned long long y = (unsigned long long)ceil(sqrtl((long double)tmp));
		pixels += y;
	}

	pixels %= k;
	MPI_Reduce(&pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		global_pixels %= k;
		printf("%llu\n", (4 * global_pixels) % k);
	} 
	MPI_Finalize();
	return 0;
}
