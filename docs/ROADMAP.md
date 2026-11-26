# Roadmap / Next Steps

Status of the project against the course specification ([`final_project_specification.pdf`](final_project_specification.pdf)) and the plan for what comes next. Check items off as they land.

## Where we are

| Piece | Sequential | OpenMP | CUDA |
|---|---|---|---|
| Laplacian sharpen | ✅ (see issues below) | ✅ (see issues below) | ⬜ planned in [`CUDA_LAPLACIAN_SHARPEN.md`](CUDA_LAPLACIAN_SHARPEN.md) |
| Nearest-neighbor scaling | ✅ | ✅ ([plan](OMP_NEIGHBOR_SCALING.md)) | ⬜ |

## Known issues in existing code

1. **Sharpen kernel shape:** seq + OpenMP run the 8-neighbor kernel (center 9); the spec requires center 5, cross −1 (4-neighbor).
2. **Sharpen borders:** seq + OpenMP skip the outer ring of pixels, leaving uninitialized memory in the output. The spec requires zero-padding with no renormalization.
3. **OpenMP sharpen** hard-codes `schedule(static)`; the spec requires comparing static vs. dynamic for p ∈ {1, 2, 4, 8}. Needs `schedule(runtime)` + CLI args like the OpenMP scaler.
4. **Sharpen programs** hard-code the input/output paths; no CLI.
5. **README** refers to `src/openmp/` (actual: `src/omp/`) and to a `SPEC.md` that doesn't exist yet.
6. **Dataset:** the spec needs portrait + landscape images at exactly 512×512 and 1024×1024. `data/` has one 1108×1280 photo and one 64×64 test image.

## Plan (in order)

1. ⬜ **Approve** [`CUDA_LAPLACIAN_SHARPEN.md`](CUDA_LAPLACIAN_SHARPEN.md) (decisions A, B, C).
2. ⬜ Implement `src/cuda/laplacian_sharpen.cu`; check with `compute-sanitizer`.
3. ⬜ Fix sequential + OpenMP sharpen: 4-neighbor default, zero-padded borders (issues 1–2).
4. ⬜ OpenMP sharpen: `schedule(runtime)`, CLI for threads/schedule/chunk/paths (issues 3–4).
5. ⬜ Design doc + implementation for CUDA nearest-neighbor (`docs/CUDA_NEIGHBOR_SCALING.md`, `src/cuda/nearest_neighbor.cu`), 0.5× and 2.0×.
6. ⬜ Verification: a test program that runs seq / OMP / CUDA on the same buffer and compares bytes (all thread counts, schedules, block sizes, odd sizes like 513×301).
7. ⬜ Timing: warm-up + 5 runs with median/range in the CPU programs too (`omp_get_wtime`), CSV output in the README's column format.
8. ⬜ Dataset: portrait + landscape at 512² and 1024², grayscale, with provenance in `data/README.md`.
9. ⬜ Build system (`Makefile`: `sequential`, `openmp`, `cuda`, `test`, `clean`) and `.gitignore` entries for new binaries.
10. ⬜ Benchmark runner script + plots, then fill the README results and hardware tables.
11. ⬜ Written report (4–6 pages) and in-class demo with images beyond the test set.
