#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#include "../common/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../common/stb_image_write.h"

#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err_ = (call);                                         \
        if (err_ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                    __FILE__, __LINE__, cudaGetErrorString(err_));         \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

typedef enum {
    SHARPEN_4_NEIGHBOR,
    SHARPEN_8_NEIGHBOR
} SharpenMode;

/* Zero-padding: any neighbor outside the image counts as 0. */
__device__ int pixel_or_zero(
    const unsigned char *__restrict__ src,
    int width,
    int height,
    int x,
    int y
) {
    if (x < 0 || x >= width || y < 0 || y >= height) return 0;
    return src[(size_t)y * width + x];
}

/* One thread per output pixel. threadIdx.x walks along a row so a warp
   reads consecutive bytes (coalesced access). */
__global__ void laplacian_sharpen_kernel(
    const unsigned char *__restrict__ src,
    unsigned char *__restrict__ dst,
    int width,
    int height,
    SharpenMode mode
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    /* Boundary masking: extra threads in the last blocks do nothing. */
    if (x >= width || y >= height) return;

    int center = src[(size_t)y * width + x];
    int top    = pixel_or_zero(src, width, height, x,     y - 1);
    int bot    = pixel_or_zero(src, width, height, x,     y + 1);
    int left   = pixel_or_zero(src, width, height, x - 1, y);
    int right  = pixel_or_zero(src, width, height, x + 1, y);

    int new_pixel;
    if (mode == SHARPEN_8_NEIGHBOR) {
        int topl = pixel_or_zero(src, width, height, x - 1, y - 1);
        int topr = pixel_or_zero(src, width, height, x + 1, y - 1);
        int botl = pixel_or_zero(src, width, height, x - 1, y + 1);
        int botr = pixel_or_zero(src, width, height, x + 1, y + 1);

        new_pixel = (9 * center) - top - bot - left - right
                    - topl - topr - botl - botr;
    } else {
        new_pixel = (5 * center) - top - bot - left - right;
    }

    new_pixel = new_pixel < 0 ? 0 : (new_pixel > 255 ? 255 : new_pixel);

    dst[(size_t)y * width + x] = (unsigned char)new_pixel;
}

int main(int argc, char **argv) {
    const char *input_path  = argc > 1 ? argv[1] : "../../data/example.jpg";
    const char *output_path = argc > 2 ? argv[2] : "sharpened_example.jpg";
    int block_size          = argc > 3 ? atoi(argv[3]) : 16;
    int mode_arg            = argc > 4 ? atoi(argv[4]) : 4;

    if (block_size != 16 && block_size != 32) {
        fprintf(stderr, "block must be 16 or 32\n");
        return 1;
    }
    if (mode_arg != 4 && mode_arg != 8) {
        fprintf(stderr, "mode must be 4 or 8\n");
        return 1;
    }
    SharpenMode mode = mode_arg == 4 ? SHARPEN_4_NEIGHBOR : SHARPEN_8_NEIGHBOR;

    /* Create the CUDA context up front so it is not counted in the timings. */
    CUDA_CHECK(cudaFree(0));

    int width, height, original_channels;
    uint8_t *input_image = stbi_load(input_path, &width, &height, &original_channels, 1);
    if (!input_image) {
        fprintf(stderr, "Error loading image %s.\n", input_path);
        return 1;
    }

    size_t total_bytes = (size_t)width * height;

    /* Pinned host buffers: faster and more stable host<->device transfers. */
    unsigned char *h_src, *h_dst;
    CUDA_CHECK(cudaMallocHost((void **)&h_src, total_bytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_dst, total_bytes));
    memcpy(h_src, input_image, total_bytes);
    stbi_image_free(input_image);

    unsigned char *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc((void **)&d_src, total_bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_dst, total_bytes));

    dim3 block(block_size, block_size);
    dim3 grid((width + block_size - 1) / block_size,
              (height + block_size - 1) / block_size);

    /* Untimed warm-up launch: the first launch pays one-time costs
       (kernel module loading), which would otherwise inflate the timing. */
    laplacian_sharpen_kernel<<<grid, block>>>(d_src, d_dst, width, height, mode);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, after_h2d, after_kernel, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&after_h2d));
    CUDA_CHECK(cudaEventCreate(&after_kernel));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Async copies (allowed because the host buffers are pinned) let the host
       queue copy -> kernel -> copy without waiting in between. A blocking
       cudaMemcpy would leave the GPU idle while the CPU wakes up to launch
       the kernel, and that host delay would be counted as kernel time. */
    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpyAsync(d_src, h_src, total_bytes, cudaMemcpyHostToDevice, 0));
    CUDA_CHECK(cudaEventRecord(after_h2d));

    laplacian_sharpen_kernel<<<grid, block>>>(d_src, d_dst, width, height, mode);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(after_kernel));
    CUDA_CHECK(cudaMemcpyAsync(h_dst, d_dst, total_bytes, cudaMemcpyDeviceToHost, 0));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms, total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, after_h2d, after_kernel));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    printf("%dx%d, mode %d, block %dx%d, grid %ux%u\n",
           width, height, mode_arg, block_size, block_size, grid.x, grid.y);
    printf("kernel %.3f ms, with transfers %.3f ms\n", kernel_ms, total_ms);

    stbi_write_jpg(output_path, width, height, 1, h_dst, 100);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(after_h2d));
    CUDA_CHECK(cudaEventDestroy(after_kernel));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFreeHost(h_src));
    CUDA_CHECK(cudaFreeHost(h_dst));

    return 0;
}
