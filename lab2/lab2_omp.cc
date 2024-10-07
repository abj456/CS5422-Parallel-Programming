#include <assert.h>
#include <stdio.h>
#include <math.h>
// #include <omp.h>
#include "/usr/lib/gcc/x86_64-linux-gnu/12/include/omp.h"

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	
	#pragma omp parallel
	{
		unsigned long long t_pixels = 0;

		#pragma omp for schedule(guided, 1000) nowait
		for(unsigned long long x = 0; x < r; x++) 
		{
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			t_pixels += y;
		}
		t_pixels %= k;

		#pragma omp critical
		pixels = (pixels + t_pixels) % k;
	}
	printf("%llu\n", (4 * pixels) % k);
}
