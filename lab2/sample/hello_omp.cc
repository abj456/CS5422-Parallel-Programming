#include <stdio.h>
#include <unistd.h>

#include <omp.h>
// #include "/usr/lib/gcc/x86_64-linux-gnu/12/include/omp.h"

int main(int argc, char** argv) {
    int omp_threads, omp_thread;

#pragma omp parallel
    {
        omp_threads = omp_get_num_threads();
        omp_thread = omp_get_thread_num();
        printf("Hello: thread %2d/%2d\n", omp_thread, omp_threads);
    }
    
    int var1 = 10;
    #pragma omp parallel num_threads(10) 
    {
        #pragma omp lastprivate(var1)
        {
            int id = omp_get_num_threads();
            sleep(id);
            var1 = id;
        }
    }
    printf("var1: %d\n", var1);
    return 0;
}
