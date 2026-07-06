from __future__ import annotations

import time

import tensorstudio as ts


def timeit(name: str, fn, iterations: int = 5) -> None:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = (time.perf_counter() - start) / iterations
    print(f"{name}: {elapsed * 1_000:.3f} ms")


def main() -> None:
    shape = (128, 128)
    a_ts = ts.randn(shape, seed=1)
    b_ts = ts.randn(shape, seed=2)
    a_np = a_ts.numpy()
    b_np = b_ts.numpy()

    timeit("tensorstudio matmul", lambda: a_ts @ b_ts)
    timeit("numpy matmul", lambda: a_np @ b_np)


if __name__ == "__main__":
    main()
