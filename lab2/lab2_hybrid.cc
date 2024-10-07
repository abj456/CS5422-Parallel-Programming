// #include "mpi.h"
#include "/opt/software/intel/oneapi/mpi/latest/include/mpi.h"
#include "/usr/lib/gcc/x86_64-linux-gnu/12/include/omp.h"
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

	int i, mpi_rank, mpi_ranks;
	MPI_Status stat;
	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	#pragma omp parallel
	{
		unsigned long long t_pixels = 0;

		#pragma omp for schedule(guided, 10) nowait
		for(unsigned long long x = mpi_rank; x < r; x += mpi_ranks) {
			unsigned long long y = (unsigned long long)ceil(sqrtl(r_sqr - x*x));
			t_pixels += y;
		}
		t_pixels %= k;

		#pragma omp critical
		pixels = (pixels + t_pixels) % k;
	}
	
	MPI_Reduce(&pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

	if (mpi_rank == 0) {
		global_pixels %= k;
		printf("%llu\n", (4 * global_pixels) % k);
	} 
	MPI_Finalize();
	return 0;
}
