# Parallel Image Processing: OpenMP vs. CUDA

**CDA 4125: Concepts of Parallel and Distributed Processing, Final Project**
Professor: Dr. Jose Ayerdis | Keiser University Latin American Campus, Software Engineering

An empirical study of the computational efficiency, scaling behavior, and synchronization overhead of two parallel programming paradigms, **OpenMP** (shared-memory CPU threads) and **CUDA** (massively parallel GPU execution), applied to two image-processing kernels: a 3×3 Laplacian sharpen stencil and a nearest-neighbor resampler.

> **Status:** 🚧 In progress. Items marked `TODO` are placeholders to fill in as the project advances. Delete this note when finished.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Team](#team)
3. [Chosen Operations and Mechanisms](#chosen-operations-and-mechanisms)
4. [Repository Structure](#repository-structure)
5. [Requirements](#requirements)
6. [Build Instructions](#build-instructions)
7. [Usage](#usage)
8. [Algorithmic Specification](#algorithmic-specification)
9. [Parallel Design](#parallel-design)
10. [Benchmarking Protocol](#benchmarking-protocol)
11. [Reproducing the Results](#reproducing-the-results)
12. [Testing and Verification](#testing-and-verification)
13. [Results Summary](#results-summary)
14. [Hardware and Software Environment](#hardware-and-software-environment)
15. [Known Limitations](#known-limitations)
16. [Documentation](#documentation)
17. [Acknowledgments and Third-Party Code](#acknowledgments-and-third-party-code)

---

## Project Overview

Image workloads exhibit distinct memory and arithmetic patterns. This project compares a sequential baseline against two parallel implementations on:

| Kernel type | Operation | Access pattern |
|---|---|---|
| Filtering | 3×3 Laplacian sharpen | Stencil / neighborhood convolution (9-point read, 5 non-zero weights) |
| Scaling | Nearest-neighbor resampling | Coordinate mapping with a single-pixel gather, no interpolation arithmetic |

Both operations are implemented sequentially, then parallelized with OpenMP and CUDA and benchmarked across thread counts, GPU block sizes, image sizes, and scheduling policies to study speedup, efficiency, cache behavior, and host-device transfer cost.

**Research questions**

- How does OpenMP speedup scale with thread count (p ∈ {1, 2, 4, 8}), and how do static and dynamic scheduling compare?
- How much of the CUDA advantage survives once host-device transfers are included (kernel-only vs. transfer-inclusive time)?
- How do 16×16 and 32×32 thread blocks compare on each kernel?
- How do small (512×512) and medium (1024×1024) images differ in scaling behavior (launch overhead vs. cache capacity)?
- Why do the filter and the scaler behave differently (stencil neighbor reuse vs. low-arithmetic gather; memory-bound behavior)?

---

## Team

| Name | Role | GitHub |
|---|---|---|
| TODO | Lead / integration | @TODO |
| TODO | Baseline and verification | @TODO |
| TODO | OpenMP implementation | @TODO |
| TODO | CUDA implementation | @TODO |
| TODO | Benchmarking and analysis | @TODO |

---

## Chosen Operations and Mechanisms

| Category | Selection |
|---|---|
| Filter | 3×3 Laplacian sharpen (center 5, cross −1) |
| Scaler | Nearest-neighbor resampling |
| Mechanism A | OpenMP (static vs. dynamic scheduling, p ∈ {1, 2, 4, 8}) |
| Mechanism B | CUDA (2D grid, 16×16 and 32×32 thread blocks, global memory, transfer profiling) |
| Language | TODO: C / C++ (standard and version); CUDA C++ for device code |

---

## Repository Structure

```
.
├── README.md              This file
├── SPEC.md                Software specification
├── Makefile               Build targets (TODO: or CMakeLists.txt)
├── src/
│   ├── common/            Image struct, I/O wrapper, timing utilities, CLI parsing
│   ├── sequential/        Baseline filter and scaler
│   ├── openmp/            OpenMP filter and scaler
│   └── cuda/              CUDA kernels, host code, transfer/timing helpers
├── tests/                 Verification harness and expected outputs
├── data/                  Benchmark images (see data/README.md for provenance)
├── scripts/               Benchmark runner, plotting, environment capture
├── results/
│   ├── raw/               Raw CSV timing data
│   └── figures/           Generated plots
├── logs/                  Dated experiment log
└── report/                Final technical report (PDF)
```

---

## Requirements

| Dependency | Version | Notes |
|---|---|---|
| C/C++ compiler | TODO (e.g., GCC ≥ 11) | Must support OpenMP |
| NVIDIA GPU | TODO (compute capability) | Required for the CUDA build |
| NVIDIA driver | TODO | Must match the CUDA toolkit |
| CUDA Toolkit (`nvcc`) | TODO (e.g., 12.x) | Required for the CUDA build |
| Make / CMake | TODO | Build system |
| Python 3 | TODO (≥ 3.9) | Plotting and benchmark post-processing only |
| Python packages | `numpy`, `pandas`, `matplotlib` | See `scripts/requirements.txt` |
| stb_image / stb_image_write | Bundled | Single-header image I/O (third-party; see [Acknowledgments](#acknowledgments-and-third-party-code)) |

The parallel mechanisms (row decomposition, scheduling, kernel design, thread-block mapping) are implemented by the team. Third-party libraries are used only for image file I/O.

---

## Build Instructions

```bash
# Clone
git clone https://github.com/<org>/<repo>.git
cd <repo>

# Build everything (TODO: confirm targets once the Makefile exists)
make all

# Build individual targets
make sequential
make openmp
make cuda          # requires nvcc and an NVIDIA GPU

# Clean
make clean
```

Compiler flags used for all reported results: `TODO` (e.g., `-O2` or `-O3`, `-fopenmp` for host code; `-arch=sm_XX` for CUDA). The same host optimization level is used for the baseline and OpenMP builds.

---

## Usage

> TODO: update the executable name and flags to match the final CLI.

```bash
./bin/imgproc --op <filter|scale> \
              --impl <seq|omp|cuda> \
              --threads <p> \
              --schedule <static|dynamic> \
              --block <16|32> \
              --scale <factor> \
              --input <path> \
              --output <path>
```

| Flag | Description | Applies to |
|---|---|---|
| `--op` | Operation: `filter` or `scale` | All |
| `--impl` | Implementation: `seq`, `omp`, `cuda` | All |
| `--threads` | Worker thread count (1, 2, 4, 8, or any positive integer) | `omp` |
| `--schedule` | OpenMP loop schedule: `static` or `dynamic` | `omp` |
| `--chunk` | Dynamic schedule chunk size (optional) | `omp` |
| `--block` | CUDA thread block edge: `16` (16×16) or `32` (32×32) | `cuda` |
| `--scale` | Scale factor (e.g., `0.5`, `2.0`, or any positive real) | `scale` |
| `--input` | Input image path (any size, including non-square and odd dimensions) | All |
| `--output` | Output image path | All |

**Examples:** filter with 4 OpenMP threads and static scheduling, upscale 2× with 16×16 CUDA blocks, run the sequential baseline. TODO: add exact command lines once the CLI is finalized.

**Input handling:** images are loaded and converted to single-channel 8-bit grayscale. Disk I/O and format conversion are excluded from all timed regions.

---

## Algorithmic Specification

### Filtering (3×3 Laplacian sharpen)

- **Kernel:** center weight 5, the four cross neighbors −1, corners 0:

  ```
   0  -1   0
  -1   5  -1
   0  -1   0
  ```
- **Borders:** zero-padding, so `I(x,y) = 0` for any out-of-bound index
- **No renormalization** of border weights (explicitly forbidden by the assignment)
- **Output size:** identical to input
- **Output range:** the kernel can produce negative values and values above 255, so results are **saturated to `[0, 255]`** (computed in a signed integer type before clamping; no wrap-around)

### Scaling (nearest-neighbor)

- **Scale factors:** arbitrary positive factors; evaluated at **0.5×** (downsample) and **2.0×** (upsample)
- **Output size:** `round(scale × input)` per dimension (TODO: document the exact rounding rule)
- **Coordinate mapping:** TODO: document the convention used (e.g., `src = floor(dst / scale)` or the pixel-center variant `src = floor((dst + 0.5) / scale)`)
- **Out-of-bound coordinates:** clamped to the nearest edge pixel
- **Value:** the single nearest source pixel is copied (no weighting)

Full details, data structures, and interface definitions are in [`SPEC.md`](SPEC.md).

---

## Parallel Design

### Sequential baseline

A single-threaded reference implementation. It defines the correct output for all verification and the `T_seq` used in all speedup calculations.

### OpenMP

- Outer-loop row parallelization with `parallel for`
- Schedules compared: **static** vs. **dynamic** (chunk size TODO)
- Thread counts: p ∈ {1, 2, 4, 8}
- For the scaler, rows are distributed over the **output** image

### CUDA

- 2D execution grid, with one thread per output pixel
- Thread blocks: **16×16** and **32×32** compared
- Grid dimensions computed with ceiling division, so image sizes that are not multiples of the block size are covered
- **Explicit boundary masking:** threads outside the output image exit without reading or writing; the filter applies zero-padding for out-of-bound neighbor reads
- Global DRAM access (no shared-memory tiling, per the project scope)
- Host-device transfer profiling: input copy (H2D), kernel, and output copy (D2H) timed separately
- Two reported times per run: **kernel-only** and **transfer-inclusive**

### Correctness safeguards

- Each thread writes a distinct output pixel (or rows are disjoint across OpenMP threads), so there are no data races
- The input is read-only and shared
- OpenMP row boundaries are chosen to avoid false sharing on output cache lines (TODO: document any padding or alignment decisions)
- CUDA error checking after every API call and kernel launch, with device synchronization before stopping the timer
- No locks in the hot path, so there is no deadlock potential

---

## Benchmarking Protocol

| Parameter | Value |
|---|---|
| Photo classes | 2 (portrait, landscape) |
| Resolutions | 512×512 (small), 1024×1024 (medium) |
| Operations | Filter; scale 0.5×; scale 2.0× |
| OpenMP configurations | p ∈ {1, 2, 4, 8} × {static, dynamic} |
| CUDA configurations | Block 16×16 and 32×32 |
| Warm-up | 1 untimed run per configuration |
| Recorded repetitions | 5 per configuration |
| Reported statistics | Median and range (min–max) |
| Timing boundaries | Compute only; excludes disk I/O and display/format conversion |
| CPU timer | TODO (e.g., `omp_get_wtime` or `clock_gettime(CLOCK_MONOTONIC)`) |
| GPU timer | TODO (e.g., CUDA events; kernel-only and transfer-inclusive) |

**Derived metrics**

- Speedup: `S = T_seq / T_parallel`
- OpenMP efficiency: `E(p) = S(p) / p`
- CUDA speedup is reported twice, using kernel-only time and transfer-inclusive time

Raw per-run timings are stored in `results/raw/` as CSV with the columns: `image, resolution, operation, implementation, schedule, threads, block, timing_mode, run, time_ms`.

---

## Reproducing the Results

```bash
# 1. Build
make all

# 2. Verify correctness before timing anything
make test

# 3. Run the full benchmark matrix (TODO: confirm the script name)
./scripts/run_benchmarks.sh

# 4. Generate tables and figures
python3 scripts/plot_results.py results/raw/ results/figures/
```

**For stable timings:**

- Close other applications and avoid running on battery power
- Use a fixed CPU governor/frequency where possible; for the GPU, note the power and clock state
- Record the machine details below with every run set
- Do not discard runs silently. Re-run and log any anomalous configuration in `logs/`

---

## Testing and Verification

Run with `make test`. The suite covers:

| Test | Purpose |
|---|---|
| Hand-computed tiny arrays (e.g., 4×4) | Verify kernel arithmetic exactly |
| Border and corner pixels | Confirm zero-padding with no weight renormalization |
| Saturation checks | Confirm sharpened values are clamped to `[0, 255]`, not wrapped or truncated |
| Scaling at 0.5× and 2.0× | Verify output dimensions and nearest-neighbor pixel selection |
| Edge clamping | Verify out-of-range source coordinates map to edge pixels |
| Parallel vs. sequential comparison | Pixel-by-pixel equality for OpenMP (every thread count and schedule) and CUDA (both block sizes) |
| Odd and non-square dimensions | Verify decomposition with remainders (e.g., 513×301) |
| Dimensions not divisible by block size | Verify CUDA boundary masking |
| Thread count > rows | Verify OpenMP behavior with more threads than rows |

Optional tooling (TODO: confirm used): ThreadSanitizer for OpenMP race checks, `compute-sanitizer` for CUDA memory and race errors, `perf` for CPU cache analysis, Nsight for GPU profiling.

---

## Results Summary

> TODO: fill in after benchmarking. Include a headline table and link the figures.

**OpenMP (median speedup vs. sequential)**

| Operation | Image | Schedule | p=1 | p=2 | p=4 | p=8 |
|---|---|---|---|---|---|---|
| Filter | 512² | static | TODO | TODO | TODO | TODO |
| Filter | 512² | dynamic | TODO | TODO | TODO | TODO |
| Filter | 1024² | static | TODO | TODO | TODO | TODO |
| ... | ... | ... | ... | ... | ... | ... |

**CUDA (median time and speedup vs. sequential)**

| Operation | Image | Block | Kernel-only | Transfer-inclusive | Speedup (kernel) | Speedup (total) |
|---|---|---|---|---|---|---|
| Filter | 512² | 16×16 | TODO | TODO | TODO | TODO |
| Filter | 512² | 32×32 | TODO | TODO | TODO | TODO |
| ... | ... | ... | ... | ... | ... | ... |

*(Ranges are in `results/`.)*

Figures: `results/figures/` (speedup and efficiency curves, static vs. dynamic comparison, CUDA kernel-only vs. transfer-inclusive).

**Key findings:** TODO (2 to 4 bullet points).

---

## Hardware and Software Environment

| Item | Value |
|---|---|
| CPU model | TODO |
| Physical cores / threads | TODO |
| Base / boost clock | TODO |
| L1 / L2 / L3 cache | TODO |
| RAM | TODO |
| GPU model | TODO |
| GPU memory and compute capability | TODO |
| PCIe generation / lanes | TODO |
| NVIDIA driver / CUDA version | TODO |
| OS and version | TODO |
| Compiler(s) and version | TODO |
| Compiler flags | TODO |
| Date of benchmark runs | TODO |

---

## Known Limitations

- TODO: e.g., fewer physical cores than the p = 8 configuration (oversubscription effects)
- Timings are specific to the hardware above; CUDA results depend heavily on the GPU and PCIe link
- With images this small (0.26 to 1.05 MP), GPU launch and transfer overhead can dominate the kernel time
- Only single-channel (grayscale) images are supported, per the project scope
- TODO: any rounding or coordinate-convention caveats

---

## Documentation

| Document | Location |
|---|---|
| Software specification | [`SPEC.md`](SPEC.md) |
| Final technical report (4 to 6 pages) | `report/` |
| Experiment log | `logs/` |
| Dataset provenance and licenses | `data/README.md` |

---

## Acknowledgments and Third-Party Code

- **stb_image / stb_image_write** (public domain / MIT): used for image I/O only
- **NVIDIA CUDA Toolkit**: runtime and compiler for the GPU implementation
- Benchmark images: TODO: list sources and licenses in `data/README.md`
- Course: CDA 4125, Dr. Jose Ayerdis, Keiser University Latin American Campus (Software Engineering), TODO: term

All parallel decomposition, scheduling, and kernel logic is original team work.
