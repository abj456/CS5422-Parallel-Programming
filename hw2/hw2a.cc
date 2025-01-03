/* pthread version */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <nvtx3/nvToolsExt.h>
// #include </opt/software/nsys-2024.5.1/target-linux-x64/nvtx/include/nvtx3/nvToolsExt.h>


void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

typedef struct ThreadData {
    int iters;
    double left;
    double right;
    double lower;
    double upper;
    int width;
    int height;
    int* image;
    int t;
    int ncpus;
} thread_data;

void* mandelbrot_calc(void* arguments) {
    thread_data *t_data = (thread_data*)arguments;
    char msg[40];
    sprintf(msg, "mandelbrot_calc() thread %d computing", t_data->t);
    nvtxRangePush(msg);
    
    double y0_base = ((t_data->upper - t_data->lower) / t_data->height);
    double x0_base = ((t_data->right - t_data->left) / t_data->width);

    for (int j = t_data->t; j < t_data->height; j += t_data->ncpus) {
        double y0 = j * y0_base + t_data->lower;
        for (int i = 0; i < t_data->width; ++i) {
            double x0 = i * x0_base + t_data->left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < t_data->iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            t_data->image[j * t_data->width + i] = repeats;
        }
    }
    nvtxRangePop();
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    nvtxRangePush("main start");
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int ncpus = CPU_COUNT(&cpu_set);

    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* mandelbrot set */
    pthread_t threads[ncpus];
    thread_data t_datas[ncpus];
    int rc;
    for(int t = 0; t < ncpus; t++) {
        t_datas[t] = (thread_data){
            .iters = iters,
            .left = left,
            .right = right,
            .lower = lower,
            .upper = upper,
            .width = width,
            .height = height,
            .image = image,
            .t = t,
            .ncpus = ncpus
        };
        rc = pthread_create(&threads[t], NULL, mandelbrot_calc, (void*)&t_datas[t]);
    }

    for(int t = 0; t < ncpus; t++) {
        pthread_join(threads[t], NULL);
    }
    nvtxRangePop();
    // for (int j = 0; j < height; ++j) {
    //     double y0 = j * ((upper - lower) / height) + lower;
    //     for (int i = 0; i < width; ++i) {
    //         double x0 = i * ((right - left) / width) + left;
    //         int repeats = 0;
    //         double x = 0;
    //         double y = 0;
    //         double length_squared = 0;
    //         while (repeats < iters && length_squared < 4) {
    //             double temp = x * x - y * y + x0;
    //             y = 2 * x * y + y0;
    //             x = temp;
    //             length_squared = x * x + y * y;
    //             ++repeats;
    //         }
    //         image[j * width + i] = repeats;
    //     }
    // }

    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);
}
