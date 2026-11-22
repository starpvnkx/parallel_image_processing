# OpenMP Nearest-Neighbor Scaling: Design Plan

This document describes how `src/omp/nearest_neighbor.c` will be built. It is the OpenMP version of the sequential scaler in `src/sequential/nearest_neighbor.c` (algorithm explained in [`NEIGHBOR_SCALING.md`](NEIGHBOR_SCALING.md)).

**Goal:** produce **pixel-for-pixel the same output** as the sequential version, faster, and let the benchmark vary the thread count and the schedule without recompiling.

---

## 1. File layout

The file keeps the same structure as the sequential version so the two are easy to compare side by side.

```
src/omp/nearest_neighbor.c
├── includes (stdio, stdlib, stdint, string, omp.h, stb_image, stb_image_write)
├── nearest_neighbor_openmp()      the parallel kernel; also builds the lookup list
├── parse_schedule()               turns "static" / "dynamic" into an omp_sched_t
└── main()                         reads arguments, sets threads and schedule,
                                   loads the image, runs the kernel, writes the output
```

The file contains only the OpenMP version. Timing and checking against the sequential output are left for later (section 5).

---

## 2. Command-line interface

Positional arguments, extending the ones the sequential program already takes:

```
./parallel_upscale [input] [output] [scale] [threads] [schedule] [chunk]
```

| Arg | Default | Meaning |
|---|---|---|
| `input` | `../../data/example4upscale.jpg` | Input image (must be square, same as the sequential version) |
| `output` | `upscaled_example.jpg` | Output path |
| `scale` | `2.0` | Any positive factor (`0.5`, `2.0`, ...) |
| `threads` | `4` | Number of OpenMP threads (`1`, `2`, `4`, `8`, ...) |
| `schedule` | `static` | `static` or `dynamic` |
| `chunk` | `0` | Chunk size; `0` uses OpenMP's default for that schedule |

**Build:**

```bash
gcc -O2 -fopenmp nearest_neighbor.c -o parallel_upscale -lm
```

---

## 3. Parallelization method

### 3.1 What gets parallelized: the output rows

The work is the "fill the new image" loop (Part 3 of the pseudocode). Each output pixel is written exactly once, and nothing depends on any other output pixel, so the rows can be split across threads safely.

```c
#pragma omp for schedule(runtime)        /* inside the parallel region from 3.3 */
for (int y = 0; y < dst_size; ++y) {
    ...fill one output row...
}
```

- **Split over output rows, not input rows.** The number of output rows sets the amount of work. When upscaling 2×, there are twice as many output rows as input rows.
- **Only the outer loop is split.** One row of a 512×512 or larger image is enough work for one task. Using `collapse(2)` to split the inner loop too would add a division and a remainder per pixel to recover `x` and `y`, which costs more than it gains for a loop this cheap.
- **`default(none)`** forces every variable to be listed as shared or private, which is the style already used in `src/omp/laplacian_sharpen.c` and catches accidental sharing at compile time.

### 3.2 Static vs. dynamic scheduling without recompiling

`schedule(runtime)` lets `main()` choose the schedule when the program starts:

```c
omp_set_num_threads(threads);
omp_set_schedule(parse_schedule(schedule_name), chunk);
```

One binary covers every combination in the benchmark (p ∈ {1, 2, 4, 8} × {static, dynamic}).

What to expect: every row costs about the same, so **static should match or beat dynamic**. Dynamic scheduling hands out rows from a shared counter, which adds a small overhead per chunk and doesn't help when rows are equal work. Showing this is one of the project's research questions.

### 3.3 The lookup list (also parallel)

The lookup list is built **inside the same parallel region** as the fill loop, so the threads are started only once:

```c
#pragma omp parallel default(none) shared(...)
{
    #pragma omp for schedule(static)
    for (int i = 0; i < dst_size; i++) source_map[i] = ...;
    /* implicit barrier: the whole list is finished before any row is filled */

    #pragma omp for schedule(runtime)
    for (int y = 0; y < dst_size; ++y) { ...fill rows... }
}
```

Each entry depends only on its own index, so the entries can be computed in any order. The barrier at the end of the first `omp for` guarantees no thread reads the list before it is complete. After that the list is read-only, so sharing it is safe.

Note that the list is small (`dst_size` integers, such as 2048 for a 2× upscale of 1024²), so the gain is modest. Putting it in the same region keeps the cost low: one extra barrier instead of a second fork and join.

### 3.4 Correctness: no races, no locks

- Each thread writes only its own rows of `dst`. No two threads write the same byte.
- `src` and `source_map` are read-only.
- No locks, atomics, or critical sections, so there's no contention and no risk of deadlock.
- **False sharing** (two threads writing to the same 64-byte cache line) can only happen where one thread's block of rows ends and the next one begins. A row is at least `dst_size × channels` bytes (≥ 256 bytes at the test sizes), so this affects at most one cache line per block boundary and is negligible.

---

## 4. Optimizations applied

| # | Optimization | Why |
|---|---|---|
| 1 | **Look up the source row once per output row.** Compute `src_row = src + source_map[y] * src_size * channels` and `dst_row = dst + y * dst_size * channels` outside the inner loop. | Removes a multiplication from every pixel; the inner loop only adds offsets. |
| 2 | **Fast path for grayscale (`channels == 1`).** Use a simple `dst_row[x] = src_row[source_map[x]]` loop instead of the per-channel inner loop. | Most benchmark images are single-channel. Removing the third loop makes the inner loop tighter and easier for the compiler to optimize. |
| 3 | **`const` and `restrict` on `src`, `dst`, and `source_map`.** | Tells the compiler the buffers don't overlap, so it can keep values in registers instead of reloading them. |
| 4 | **`size_t` for all index math.** | Prevents overflow on large outputs (a 2× upscale of a 3-channel 8K image is more than 2³¹ bytes). |
| 5 | **Parallel lookup list** in the same parallel region (section 3.3). | Splits the list work across threads without paying for a second thread start-up. |
| 6 | **Copy duplicate rows with `memcpy`** (section 4.1). | When upscaling, several output rows come from the same source row. Building that row once and copying it is much cheaper than gathering it pixel by pixel again. |

### 4.1 Duplicate-row copying without races

When upscaling 2×, the lookup list for rows looks like `[0, 0, 1, 1, 2, 2, ...]`. Consecutive output rows that share a source row form a **run**.

The obvious approach, "if row `y` matches row `y-1`, copy row `y-1`", is a **data race**: row `y-1` might belong to another thread that hasn't written it yet.

Instead, **the first row of each run owns the whole run**:

```c
for (int y = 0; y < dst_size; ++y) {            /* parallel */
    int from_y = source_map[y];
    if (y > 0 && source_map[y - 1] == from_y)
        continue;                                /* not a run start: its owner fills it */

    ...gather row y pixel by pixel...

    for (int r = y + 1; r < dst_size && source_map[r] == from_y; ++r)
        memcpy(row r, row y, row_bytes);         /* same thread, row y is already done */
}
```

- Each output row is still written by exactly one thread, the one that owns the run's first row, so there is no race no matter how the schedule splits the rows.
- The `memcpy` reads a row the same thread just wrote, so no synchronization is needed.
- When downscaling (0.5×), every run has length 1, and the loop behaves exactly like the plain version.
- **Load balance:** run-start iterations do the work and the others skip, but runs are evenly spaced for a fixed scale factor, so static scheduling still gives each thread about the same amount of work.

**Note for benchmarking later:** the sequential baseline doesn't copy rows, so part of the OpenMP speedup will come from this algorithmic change, not only from threads. To keep the two effects apart, also measure the OpenMP version at 1 thread: `T_seq / T_omp(p=1)` is the gain from the optimizations alone, and `T_omp(1) / T_omp(p)` is the gain from threads alone.

**Left out for now:** SIMD pragmas (`omp simd`). The inner loop reads from scattered positions in the source (a gather), which vectorizes poorly on most CPUs. The compiler can still auto-vectorize it at `-O2`/`-O3` if it finds a way.

---

## 5. What `main()` does

1. Read and check the arguments (scale > 0, threads ≥ 1, chunk ≥ 0, schedule is `static` or `dynamic`).
2. Call `omp_set_num_threads(threads)` and `omp_set_schedule(schedule, chunk)`.
3. Load the image and check that it is square.
4. Allocate the output buffer and run `nearest_neighbor_openmp()` once.
5. Print a one-line summary of the configuration, write the output image, and free everything.

Example output:

```
size 1024 -> 2048, channels 1, threads 4, schedule static, chunk 0
```

**Not included yet:** timing (warm-up, 5 timed runs, median) and the byte-for-byte check against the sequential output. These will be added later, for example in a separate benchmark or test program, following the protocol in the README.

---

## 6. Scope and limitations

- **Square images only**, the same as the sequential version, so both versions accept exactly the same inputs. Supporting non-square images (separate row and column lookup lists) should be done in both files together later.
- Output size is `(int)(size × scale)` with a minimum of 1, the same rule as the sequential version.
- JPEG output is lossy, so when the correctness check is added, it should compare the in-memory buffers, not the saved files.
