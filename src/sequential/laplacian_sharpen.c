#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"

typedef enum {
    SHARPEN_4_NEIGHBOR,
    SHARPEN_8_NEIGHBOR
} SharpenMode;

void laplacian_sharpen_sequential(
    const unsigned char *src, 
    unsigned char *dst,
    int width, 
    int height,
    SharpenMode mode
) {
    int center_weight = mode == SHARPEN_4_NEIGHBOR ? 5 : 9;
    for(int y = 1; y<height-1; ++y) {
        for (int x = 1; x<width-1; ++x) {
            int new_pixel;
            int center_index = (y * width) + x;
            int top_index = ((y - 1) * width) + x;
            int bot_index = ((y + 1) * width) + x;
            int left_index = (y * width) + (x - 1);
            int right_index = (y * width) + (x + 1);

            int center = src[center_index];
            int top    = src[top_index];
            int bot    = src[bot_index];
            int left   = src[left_index];
            int right  = src[right_index];
            
            if (mode == SHARPEN_8_NEIGHBOR) {
                int topr = src[((y - 1) * width) + (x + 1)];
                int topl = src[((y - 1) * width) + (x - 1)];
                int botr = src[((y + 1) * width) + (x + 1)];
                int botl = src[((y + 1) * width) + (x - 1)];

                new_pixel = (center_weight * center) - top - bot - left - right
                            - topr - topl - botr - botl;
            } else {
                new_pixel = (center_weight * center) - top - bot - left - right;
            }

            new_pixel = new_pixel < 0 ? 0 : (new_pixel > 255 ? 255 : new_pixel);

            dst[center_index] = (unsigned char)new_pixel;
        }
    }
}

int main(int argc, char** argv) {
    int width, height, original_channels;

    uint8_t *input_image = stbi_load(
        "../../data/example.jpg",
        &width,
        &height,
        &original_channels,
        1
    );

    if (!input_image) {
        printf("Error loading image.\n");
        return 1;
    }

    size_t total_bytes = (size_t)width * height;
    uint8_t *output_image = (uint8_t *)malloc(total_bytes);

    laplacian_sharpen_sequential(
        input_image,
        output_image,
        width,
        height,
        SHARPEN_4_NEIGHBOR
    );
    stbi_write_jpg("sharpened_example.jpg", width, height, 1, output_image, 100);

    stbi_image_free(input_image);
    free(output_image);

    return 0;
}