# Benchmarks

TensorStudio includes simple local benchmark scripts under `benchmarks/`.

## Goals

The benchmarks are meant to provide rough local context, not marketing claims.
Do not claim TensorStudio is faster than NumPy or other frameworks unless a
carefully controlled benchmark proves it.

## Run Elementwise Benchmarks

```bash
python benchmark_all.py
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

This compares TensorStudio 2D matrix multiplication with NumPy and, when
installed, PyTorch on the same shapes.

## Run Convolution Benchmarks

```bash
python benchmarks/bench_conv2d.py
```

This compares the native CPU NCHW `conv2d` kernel with a simple NumPy reference
loop and, when installed, PyTorch, TensorFlow, and JAX CPU convolution. The
NumPy baseline is intentionally simple; PyTorch and other mature libraries use
far more optimized CPU kernels.

## Run Pooling Benchmarks

```bash
python benchmarks/bench_pooling.py
```

This compares native CPU NCHW `max_pool2d` and `avg_pool2d` with a simple
NumPy reference and, when installed, PyTorch, TensorFlow, and JAX CPU
equivalents where available.

## Interpreting Results

Expect NumPy and PyTorch to be faster for many medium and large operations
because they use highly optimized native kernels. TensorStudio `1.5.1` still
favors clarity, portability, and a compact C++ implementation over full kernel
library performance.

The current benchmark report records local NumPy, TensorFlow, PyTorch, and JAX
win/loss counts when those libraries are installed. Detailed tables include
explicit win columns and a fastest-library column for each case.
On the Windows CPython 3.10 run checked into `benchmarks/results.md`,
TensorStudio wins many small PyTorch CPU eager cases, but loses larger matrix
multiplication, convolution against PyTorch CPU, larger transcendental
activations, and larger autograd cases. Those losses are expected until
TensorStudio has BLAS, SIMD, and a real kernel scheduler.

Useful benchmark notes:

- Run multiple times.
- Close other heavy processes.
- Record Python version and CPU.
- Record TensorStudio version.
- Read win columns together with speedup columns. A win applies only to that
  exact operation, shape, dtype, Python version, and CPU.
- Do not compare debug builds with release builds.

## Future Benchmark Work

- Add CI benchmark trend recording.
- Add broader shape sweep benchmarks.
- Add CI smoke benchmarks.
- Add memory allocation measurements.
