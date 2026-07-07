from __future__ import annotations

import tensorstudio as ts


def main() -> None:
    x = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = ts.ones((2, 2))

    print("x + y =", (x + y).tolist())
    print("x * 2 =", (x * 2).tolist())
    print("x @ y =", (x @ y).tolist())
    print("mean(x) =", x.mean().item())


if __name__ == "__main__":
    main()
