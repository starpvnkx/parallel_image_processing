/* OpenMP nearest-neighbor scaler.
 * Design notes: OMP_NEIGHBOR_SCALING.md at the repository root.
 *
 * Build: gcc -O2 -fopenmp nearest_neighbor.c -o parallel_upscale -lm
 * Usage: ./parallel_upscale [input] [output] [scale] [threads] [schedule] [chunk]
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"

/* Square images only: width == height, so rows and cols scale the
 * same way and one lookup list serves both directions.
 *
 * src is src_size * src_size, dst is dst_size * dst_size.
 * The schedule and thread count come from the OpenMP runtime
 * (omp_set_schedule / omp_set_num_threads in main).
 * Returns 0 on success, 1 if the lookup list could not be allocated.
 */
int nearest_neighbor_openmp(
    const unsigned char *restrict src,
    unsigned char *restrict dst,
    int src_size,
    int dst_size,
    int channels
) {
    int *restrict source_map = malloc((size_t)dst_size * sizeof(int));
    if (!source_map) {
        return 1;
    }

    size_t src_stride = (size_t)src_size * channels;  // bytes per source row
    size_t dst_stride = (size_t)dst_size * channels;  // bytes per output row

    /* One parallel region: threads start once for both loops. */
    #pragma omp parallel default(none) \
            shared(src, dst, source_map, src_size, dst_size, channels, \
                   src_stride, dst_stride)
    {
        /* Lookup list: each entry depends only on its own index. */
        #pragma omp for schedule(static)
        for (int i = 0; i < dst_size; i++) {
            int answer = (int)(((long long)i * src_size) / dst_size);
            if (answer > src_size - 1) {
                answer = src_size - 1;  // never go past the last row/col
            }
            source_map[i] = answer;
        }
        /* Implicit barrier: the list is complete before any row is filled. */

        #pragma omp for schedule(runtime)
        for (int y = 0; y < dst_size; ++y) {
            int from_y = source_map[y];

            /* Not the first row of its run: the run's owner fills it. */
            if (y > 0 && source_map[y - 1] == from_y) {
                continue;
            }

            const unsigned char *restrict src_row = src + (size_t)from_y * src_stride;
            unsigned char *restrict dst_row = dst + (size_t)y * dst_stride;

            if (channels == 1) {
                for (int x = 0; x < dst_size; ++x) {
                    dst_row[x] = src_row[source_map[x]];
                }
            } else {
                for (int x = 0; x < dst_size; ++x) {
                    const unsigned char *from = src_row + (size_t)source_map[x] * channels;
                    unsigned char *to = dst_row + (size_t)x * channels;
                    for (int c = 0; c < channels; ++c) {
                        to[c] = from[c];
                    }
                }
            }

            /* The rest of the run copies the row this thread just built. */
            for (int r = y + 1; r < dst_size && source_map[r] == from_y; ++r) {
                memcpy(dst + (size_t)r * dst_stride, dst_row, dst_stride);
            }
        }
    }

    free(source_map);
    return 0;
}

/* Returns 0 on success, 1 for an unknown schedule name. */
static int parse_schedule(const char *name, omp_sched_t *kind)
{
    if (strcmp(name, "static") == 0) {
        *kind = omp_sched_static;
    } else if (strcmp(name, "dynamic") == 0) {
        *kind = omp_sched_dynamic;
    } else {
        return 1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *input_path    = argc > 1 ? argv[1] : "../../data/example4upscale.jpg";
    const char *output_path   = argc > 2 ? argv[2] : "upscaled_example.jpg";
    double scale              = argc > 3 ? atof(argv[3]) : 2.0;
    int threads               = argc > 4 ? atoi(argv[4]) : 4;
    const char *schedule_name = argc > 5 ? argv[5] : "static";
    int chunk                 = argc > 6 ? atoi(argv[6]) : 0;

    if (scale <= 0.0) {
        printf("Scale must be greater than 0.\n");
        return 1;
    }
    if (threads < 1) {
        printf("Threads must be at least 1.\n");
        return 1;
    }
    if (chunk < 0) {
        printf("Chunk must be 0 (default) or positive.\n");
        return 1;
    }

    omp_sched_t schedule;
    if (parse_schedule(schedule_name, &schedule)) {
        printf("Schedule must be 'static' or 'dynamic', got '%s'.\n", schedule_name);
        return 1;
    }

    omp_set_num_threads(threads);
    omp_set_schedule(schedule, chunk);  // chunk 0 = OpenMP default

    int width, height, channels;

    /* Last argument 0 = keep the file's own channel count */
    uint8_t *input_image = stbi_load(input_path, &width, &height, &channels, 0);
    if (!input_image) {
        printf("Error loading image: %s\n", stbi_failure_reason());
        return 1;
    }

    if (width != height) {
        printf("Image must be square, but it is %d x %d.\n", width, height);
        stbi_image_free(input_image);
        return 1;
    }

    int size     = width;
    int new_size = (int)(size * scale);
    if (new_size < 1) new_size = 1;

    size_t total_bytes = (size_t)new_size * new_size * channels;
    uint8_t *output_image = malloc(total_bytes);
    if (!output_image) {
        printf("Not enough memory for the output image.\n");
        stbi_image_free(input_image);
        return 1;
    }

    if (nearest_neighbor_openmp(input_image, output_image,
                                size, new_size, channels)) {
        printf("Not enough memory for the lookup list.\n");
        stbi_image_free(input_image);
        free(output_image);
        return 1;
    }

    printf("size %d -> %d, channels %d, threads %d, schedule %s, chunk %d\n",
           size, new_size, channels, threads, schedule_name, chunk);

    if (!stbi_write_jpg(output_path, new_size, new_size, channels, output_image, 100)) {
        printf("Error writing image.\n");
    }

    stbi_image_free(input_image);
    free(output_image);

    return 0;
}
