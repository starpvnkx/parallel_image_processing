#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"

/* Square images only: width == height, so rows and cols scale the
 * same way and one lookup list server both directions.
 */
static void build_source_map(int *map, int dst_size, int src_size)
{
    for (int i=0; i<dst_size; i++) {
        int answer = (int)(((long long)i * src_size) / dst_size);
        if (answer > src_size - 1) {
            answer = src_size - 1;  // never go past the last row/col
        }
        map[i] = answer;
    }
}

/* src is src_size * src_size, dst is dst_size * dst_size.
 * Returns 0 on success, 1 if the lookup list could not be allocated
 */
int nearest_neighbor_sequential(
    const unsigned char *src,
    unsigned char *dst,
    int src_size,
    int dst_size,
    int channels
) {
    int *source_map = malloc((size_t)dst_size * sizeof(int));
    if (!source_map) {
        return 1;
    }

    build_source_map(source_map, dst_size, src_size);

    for (int y=0; y<dst_size; ++y) {
        int from_y = source_map[y];      // which original row

        for (int x=0; x<dst_size; ++x) {
            int from_x = source_map[x];  // which original column

            /* dst_size for dst, src_size for src */
            size_t dst_index = ((size_t)y * dst_size + x) * channels;
            size_t src_index = ((size_t)from_y * src_size + from_x) * channels;

            for (int c=0; c<channels; ++c) {
                dst[dst_index + c] = src[src_index + c];
            }
        }
    }

    free(source_map);
    return 0;
}

int main(int argc, char **argv)
{
    const char *input_path  = argc > 1 ? argv[1] : "../../data/example4upscale.jpg";
    const char *output_path = argc > 2 ? argv[2] : "upscaled_example.jpg";
    double scale            = argc > 3 ? atof(argv[3]) : 2.0;

    if (scale <= 0.0) {
        printf("Scale must be greater than 0.\n");
        return 1;
    }

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

    if (nearest_neighbor_sequential(input_image, output_image,
                                    size, new_size, channels)) {
        printf("Not enough memory for the lookup list.\n");
        stbi_image_free(input_image);
        free(output_image);
        return 1;
    }

    if (!stbi_write_jpg(output_path, new_size, new_size, channels, output_image, 100)) {
        printf("Error writing image.\n");
    }

    stbi_image_free(input_image);
    free(output_image);

    return 0;
}
