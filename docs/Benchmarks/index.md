# Benchmarks

TensorStudio includes simple local benchmark scripts under `benchmarks/`.

## Goals

The benchmarks are meant to provide rough local context, not marketing claims.
Do not claim TensorStudio is faster than NumPy or other frameworks unless a
carefully controlled benchmark proves it.

## Run Elementwise Benchmarks

```bash
python benchmarks/bench_tensor_ops.py
```

This compares:

- elementwise add
- ReLU

against equivalent NumPy operations.

## Run Matmul Benchmarks

```bash
python benchmarks/bench_matmul.py
```

This compares TensorStudio 2D matrix multiplication with NumPy matrix
multiplication on the same shapes.

## Interpreting Results

Expect NumPy to be faster for many operations because it uses highly optimized
native kernels. TensorStudio v0.1.x favors clarity and correctness over kernel
performance.

Useful benchmark notes:

- Run multiple times.
- Close other heavy processes.
- Record Python version and CPU.
- Record TensorStudio version.
- Do not compare debug builds with release builds.

## Future Benchmark Work

- Add benchmark result recording.
- Add shape sweep benchmarks.
- Add autograd backward benchmarks.
- Add CI smoke benchmarks.
- Add memory allocation measurements.
