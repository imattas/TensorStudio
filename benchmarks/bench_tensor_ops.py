from __future__ import annotations

import time

import numpy as np
import tensorstudio as ts


def timeit(name: str, fn, iterations: int = 20) -> None:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = (time.perf_counter() - start) / iterations
    print(f"{name}: {elapsed * 1_000:.3f} ms")


def main() -> None:
    size = 50_000
    a_ts = ts.randn((size,), seed=1)
    b_ts = ts.randn((size,), seed=2)
    a_np = a_ts.numpy()
    b_np = b_ts.numpy()

    timeit("tensorstudio add", lambda: a_ts + b_ts)
    timeit("numpy add", lambda: a_np + b_np)
    timeit("tensorstudio relu", lambda: a_ts.relu())
    timeit("numpy relu", lambda: np.maximum(a_np, 0))


if __name__ == "__main__":
    main()
