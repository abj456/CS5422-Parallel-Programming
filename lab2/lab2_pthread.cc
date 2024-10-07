#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

typedef struct thread_data {
	unsigned long long x_start;
	unsigned long long x_end;
	unsigned long long r;
	unsigned long long k;
	unsigned long long return_pixels;
} thread_data;

void* pixel_cal(void* arguments) {
	thread_data *t_data = (thread_data*)arguments;
	unsigned long long x_start = t_data->x_start;
	unsigned long long x_end = t_data->x_end;
	unsigned long long r = t_data->r;
	unsigned long long k = t_data->k;
	unsigned long long pixels = 0;

	for (unsigned long long x = x_start; x < x_end; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}
	pixels %= k;
	t_data->return_pixels = pixels;
	pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long total_pixels = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	// printf("num of threads: %llu\n", ncpus);

	pthread_t threads[ncpus];
	thread_data t_datas[ncpus];
	int rc;
	for(unsigned long long t = 0; t < ncpus; t++) {
		t_datas[t] = (thread_data){.r = r, .k = k};
		t_datas[t].x_start = t * (r / ncpus);
		t_datas[t].x_end = (t == ncpus - 1) ? (r) : (t + 1) * (r / ncpus);

		rc = pthread_create(&threads[t], NULL, pixel_cal, (void*)&t_datas[t]);
	}

	for(unsigned long long t = 0; t < ncpus; t++) {
		pthread_join(threads[t], NULL);
		total_pixels += t_datas[t].return_pixels;
	}
	printf("%llu\n", (4 * total_pixels) % k);
}
