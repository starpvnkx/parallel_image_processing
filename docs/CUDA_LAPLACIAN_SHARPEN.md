# CUDA Laplacian Sharpen: Design Plan

This document describes how `src/cuda/laplacian_sharpen.cu` will be built. It is the CUDA version of the sequential filter in `src/sequential/laplacian_sharpen.c` (algorithm explained in [`LAPLACIAN_SHARPEN.md`](LAPLACIAN_SHARPEN.md)) and the GPU counterpart of `src/omp/laplacian_sharpen.c`.

**Goal:** produce **pixel-for-pixel the same output** as the sequential version, follow the CUDA rules in the project specification (2D grid, 16×16 or 32×32 blocks, explicit boundary masking, global memory, transfer profiling), and report both **kernel-only** and **transfer-inclusive** times.

> **Status:** proposal, waiting for approval. Sections marked **⚠ Decision** need a yes/no before coding starts.

---

## 0. Problems found in the current filter (must be settled first)

Reading the spec next to the existing sequential and OpenMP code turned up two mismatches. The CUDA version can't be "the same output as sequential" *and* "follow the spec" until these are resolved.

### 0.1 Kernel shape: the spec says 4-neighbor, the code runs 8-neighbor

| Source | Kernel |
|---|---|
| Project spec (PDF, §2) | "3×3 Laplacian Sharpen (**center 5, cross −1**)" → 4-neighbor |
| `README.md` | Center 5, cross −1, corners 0 → 4-neighbor |
| `src/sequential/…` and `src/omp/…` `main()` | `SHARPEN_8_NEIGHBOR` (center 9, all eight −1) |

```
spec (4-neighbor)        code today (8-neighbor)
 0  -1   0               -1  -1  -1
-1   5  -1               -1   9  -1
 0  -1   0               -1  -1  -1
```

**⚠ Decision A:** The CUDA kernel supports both modes (same `SharpenMode` enum as the other two files), but **defaults to `SHARPEN_4_NEIGHBOR`**, and the sequential and OpenMP `main()` are switched back to 4-neighbor in a separate commit so all three agree with the spec. The 8-neighbor mode stays available as an extra.

### 0.2 Borders: the spec says zero-padding, the code skips them

The spec requires: *"Apply zero-padding on external image borders (I(x,y) = 0 for out-of-bound indices). Preserve original dimensions. Renormalization of border weights is strictly forbidden."*

The current loops run `y = 1 … height-2` and `x = 1 … width-2`, so the outermost ring of pixels is **never written**. Since `output_image` comes from `malloc` (not `calloc`), those pixels contain whatever garbage was in memory. That is both a spec violation and a non-deterministic output (two runs can differ at the border, which would break any sequential-vs-parallel comparison).

Zero-padding means every pixel, including borders, is computed. A neighbor that falls outside the image counts as `0`, and the weights are **not** adjusted. Example, top-left corner `(0,0)` with the 4-neighbor kernel:

```
out(0,0) = 5·I(0,0) − I(0,−1) − I(−1,0) − I(1,0) − I(0,1)
         = 5·I(0,0) −    0    −    0    − I(1,0) − I(0,1)
```

(No renormalization means we do *not* rescale the 5 to compensate for the two missing neighbors. Corners and edges will come out brighter than the interior; that is the expected, spec-mandated result.)

**⚠ Decision B:** The CUDA kernel implements true zero-padding over the whole image, and the sequential and OpenMP versions get the same fix (separate commit, before benchmarking).

---

## 1. File layout

Same structure as the other two filter files, so the three can be read side by side.

```
src/cuda/laplacian_sharpen.cu
├── includes (stdio, stdlib, stdint, cuda_runtime.h, stb_image, stb_image_write)
├── CUDA_CHECK(call)                 macro: aborts with file/line + cudaGetErrorString
├── SharpenMode enum                 same values as the C versions
├── pixel_or_zero()      __device__  returns src[y*w+x], or 0 if (x,y) is outside
├── laplacian_sharpen_kernel()  __global__   one thread = one output pixel
├── laplacian_sharpen_cuda()     host        H2D copy → launch → D2H copy, timed with events
├── median / min / max helpers   host        for the 5 timed runs
└── main()                                   args, load image, pinned buffers, warm-up + 5 runs, write output
```

---

## 2. Command-line interface

Positional, following the style of the OpenMP nearest-neighbor plan:

```
./cuda_sharpen [input] [output] [block] [mode] [runs]
```

| Arg | Default | Meaning |
|---|---|---|
| `input` | `../../data/example.jpg` | Input image (loaded as 1-channel grayscale, any width/height) |
| `output` | `sharpened_example.jpg` | Output path |
| `block` | `16` | Thread block edge: `16` (16×16) or `32` (32×32) |
| `mode` | `4` | `4` or `8` neighbor kernel |
| `runs` | `5` | Timed repetitions after 1 untimed warm-up |

**Build** (RTX 5070 Laptop = compute capability 12.0, CUDA 13.1):

```bash
nvcc -O2 -arch=sm_120 -diag-suppress 550 laplacian_sharpen.cu -o cuda_sharpen
```

---

## 3. What changes compared to the sequential / OpenMP versions

| Aspect | Sequential / OpenMP | CUDA |
|---|---|---|
| Unit of work | One row (OpenMP) / whole image (seq) | **One pixel per thread** |
| Loops | `for y { for x { … } }` | **No loops.** The grid *is* the loop: `x = blockIdx.x·blockDim.x + threadIdx.x`, same for `y` |
| Where data lives | Host RAM | Copied to GPU global memory (H2D), result copied back (D2H) |
| Border handling | Skipped (bug, see §0.2) | `pixel_or_zero()` returns 0 for out-of-bound neighbors |
| Out-of-image threads | N/A | **Boundary mask:** `if (x >= width || y >= height) return;` |
| Errors | Only the image load is checked | Every CUDA call wrapped in `CUDA_CHECK`, plus `cudaGetLastError()` after the launch |
| Timing | None yet | CUDA events around H2D, kernel, D2H |

The per-pixel math (read 5 or 9 values, weighted sum in `int`, clamp to `[0,255]`, store as `unsigned char`) is **identical**, which is what makes the outputs bit-exact. There is no floating point anywhere, so CPU and GPU cannot round differently.

---

## 4. Pseudocode

### 4.1 Device side

```c
__device__ int pixel_or_zero(const uint8_t *src, int w, int h, int x, int y) {
    if (x < 0 || x >= w || y < 0 || y >= h) return 0;   // zero-padding
    return src[(size_t)y * w + x];
}

__global__ void laplacian_sharpen_kernel(const uint8_t *__restrict__ src,
                                         uint8_t       *__restrict__ dst,
                                         int w, int h, SharpenMode mode)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;   // column  (fast-moving → coalesced)
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row

    if (x >= w || y >= h) return;                    // explicit boundary masking

    int c = src[(size_t)y * w + x];                  // center is always in bounds
    int sum;
    if (mode == SHARPEN_4_NEIGHBOR) {
        sum = 5 * c
            - pixel_or_zero(src, w, h, x,   y-1)     // top
            - pixel_or_zero(src, w, h, x,   y+1)     // bottom
            - pixel_or_zero(src, w, h, x-1, y  )     // left
            - pixel_or_zero(src, w, h, x+1, y  );    // right
    } else {
        sum = 9 * c - (all 8 neighbors via pixel_or_zero);
    }

    sum = min(max(sum, 0), 255);                     // saturate, no wrap-around
    dst[(size_t)y * w + x] = (uint8_t)sum;
}
```

### 4.2 Host side

```
main:
    parse args, validate block ∈ {16, 32}, mode ∈ {4, 8}, runs ≥ 1
    cudaFree(0)                                     // create the CUDA context now, outside any timer
    load image with stbi_load(..., 1)               // grayscale; disk I/O not timed
    allocate pinned host buffers h_src, h_dst (cudaMallocHost), memcpy image into h_src
    allocate device buffers d_src, d_dst (cudaMalloc)   // allocation not timed
    create 4 CUDA events: start, after_h2d, after_kernel, stop

    block = dim3(B, B)
    grid  = dim3(ceil(w / B), ceil(h / B))          // covers sizes that aren't multiples of B

    repeat 1 + runs times:
        record start
        cudaMemcpy d_src ← h_src                    (H2D)
        record after_h2d
        launch kernel<<<grid, block>>>
        CUDA_CHECK(cudaGetLastError())
        record after_kernel
        cudaMemcpy h_dst ← d_dst                    (D2H)
        record stop
        cudaEventSynchronize(stop)
        if not the warm-up run:
            kernel_ms[i] = elapsed(after_h2d, after_kernel)
            total_ms[i]  = elapsed(start, stop)
            (also keep h2d_ms and d2h_ms for the report)

    print median and [min, max] of kernel_ms and total_ms
    write h_dst to output (not timed)
    free everything, destroy events
```

Example output:

```
1108x1280, mode 4, block 16x16, grid 70x80, runs 5
kernel  median 0.041 ms  range [0.039, 0.044]
total   median 0.612 ms  range [0.598, 0.640]   (h2d 0.31, d2h 0.26)
```

---

## 5. Decisions and why

| # | Decision | Why |
|---|---|---|
| 1 | **One thread per output pixel, 2D grid** | Required by the spec. The filter has no dependencies between output pixels, so it maps 1:1 onto threads with no synchronization at all. |
| 2 | **`threadIdx.x` → column, `threadIdx.y` → row** | Pixels in a row are consecutive in memory. Making `x` the fastest-varying index means the 32 threads of a warp read 32 consecutive bytes → one coalesced memory transaction instead of 32. Swapping them would be a large, easy-to-make performance bug. |
| 3 | **Block size is a runtime argument (16 or 32)** | One binary covers both spec configurations, like `schedule(runtime)` does for OpenMP. |
| 4 | **Grid uses ceiling division + boundary mask** | `example.jpg` is 1108×1280, which is not a multiple of 16 or 32. The extra threads in the last row/column of blocks must exit without touching memory, otherwise they write out of bounds. |
| 5 | **Zero-padding via a small `__device__` helper**, not a padded copy of the image | Padding the image would add a host-side copy and change the buffer size/indexing. The helper is a bounds check; only threads on the image edge take the "return 0" branch, so warp divergence is limited to the border warps (a tiny fraction of the image). |
| 6 | **Global memory only, no shared-memory tiling** | The spec lists "global DRAM access" for CUDA and the README already commits to it. Neighbor reuse still happens through the GPU's L1/L2 caches. Tiling can be added later as a clearly labelled extra experiment if we want it in the report. |
| 7 | **`int` math, clamp, then cast** | Same as the C versions: `5·255 = 1275` and `−4·255` don't fit in `uint8_t`, so we compute signed and saturate. Bit-exact with the CPU. |
| 8 | **`mode` as a normal kernel argument (not a template)** | Every thread in the launch takes the same branch, so there is no divergence cost; keeps the code close to the C versions. |
| 9 | **`const … __restrict__` and `size_t` indexing** | `__restrict__` lets the compiler route read-only loads through the read-only cache; `size_t` avoids overflow on very large images (same reasoning as the OpenMP plan). |
| 10 | **CUDA events for timing**, 4 per run | Events are timestamped on the GPU, so they measure GPU work precisely without host-side noise. Kernel-only = `after_h2d → after_kernel`; transfer-inclusive = `start → stop`. This is exactly the split the spec asks for. |
| 11 | **Warm-up + 5 runs live inside the program** | The first launch in a process pays one-time costs (module load, clock ramp-up). An external script that restarts the binary would pay them every time, so the warm-up must be in the same process. |
| 12 | **Context creation (`cudaFree(0)`), `cudaMalloc`, image load and file write are outside the timers** | The spec excludes disk I/O and conversions; allocation and context setup are one-time costs, not part of the per-image work. |
| 13 | **Pinned host memory (`cudaMallocHost`)** | Pageable memory makes the driver copy through a hidden staging buffer, which makes transfer times slower and noisier. Pinned buffers give the real PCIe transfer cost. The copy from the `stbi_load` buffer into the pinned buffer happens once, before timing. **⚠ Decision C:** pinned (recommended) vs. plain `malloc` buffers. |
| 14 | **`CUDA_CHECK` on every API call + `cudaGetLastError()` after launch** | Kernel launch errors (bad block size, etc.) are otherwise silent. Required for "implementation quality". |

---

## 6. Correctness argument (races, deadlocks, false sharing)

- Each thread writes **exactly one** byte of `dst`, at its own `(x, y)`. No two threads share an output location → no races.
- `src` is read-only during the kernel.
- No `__syncthreads()`, atomics or locks → no deadlock possible.
- False sharing is a CPU cache-coherence problem; it does not apply to GPU global-memory writes in the same way. Adjacent threads writing adjacent bytes is actually the *desired* (coalesced) pattern.
- `compute-sanitizer --tool memcheck ./cuda_sharpen` will be run once to confirm no out-of-bounds accesses (this checks decision 4).

---

## 7. What to expect in the results (hypotheses for the report)

- **Kernel time will be tiny** (tens of microseconds at 512² / 1024²) and **transfers will dominate** the transfer-inclusive time. This is the main CUDA story for small images.
- **16×16 vs 32×32:** a 32×32 block has 1024 threads. On compute capability 12.0 an SM holds at most 1536 threads, so only **one** 32×32 block fits per SM (≈67% occupancy), while **six** 16×16 blocks fit (100%). Expect 16×16 to be equal or slightly faster. (To confirm with `deviceQuery` / Nsight Compute.)
- The kernel is **memory-bound**: ~1 store and 5 loads per pixel with ~5 integer ops. Arithmetic is not the bottleneck.

---

## 8. Scope and limitations

- Grayscale only (spec: single-channel), any width/height.
- No timing of the stb load/write.
- Byte-for-byte comparison against the sequential output is a separate step (see [`ROADMAP.md`](ROADMAP.md)); compare **in-memory buffers**, not the saved JPEGs, since JPEG is lossy.
- **Not included yet:** shared-memory tiling, multiple pixels per thread, CUDA streams to overlap transfer and compute. Possible extras once the required work is done.
