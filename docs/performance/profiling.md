# Profiling

Performance work should start with measurement. TensorStudio is young enough
that guesses are often wrong: Python dispatch, allocation, broadcasting,
strided access, and math kernels can each dominate different workloads.

## Local Benchmark Entry Point

Run every benchmark group:

```bash
python benchmark_all.py
```

Run a focused section:

```bash
python benchmark_all.py --section elementwise
python benchmark_all.py --section matmul
python benchmark_all.py --section conv2d
```

Reports are written to `benchmarks/results.md`.

## What To Inspect

When a case is slow, separate these costs:

- Tensor construction and dtype conversion.
- Shape validation and broadcasting setup.
- Native kernel loop time.
- Autograd graph creation.
- Backward graph traversal.
- Python wrapper dispatch.

For tiny tensors, dispatch and allocation can dominate. For larger tensors, the
inner C++ loops and memory access pattern matter more.

## Windows Notes

On Windows, make sure benchmarks are run from the same Python version and wheel
type you plan to compare. Debug builds and editable builds can behave
differently from release wheels.

## Reporting Results

Benchmark reports must stay honest:

- Say which competitors were installed.
- Keep CPU/GPU results separate.
- Do not claim global superiority from a narrow case.
- Include losses, not only wins.
- Keep local machine details in the report.

This lets performance improvements be celebrated without turning the docs into
marketing fiction.
