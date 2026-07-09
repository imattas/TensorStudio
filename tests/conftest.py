from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = str(ROOT / "python")

sys.meta_path[:] = [
    finder for finder in sys.meta_path if type(finder).__name__ != "ScikitBuildRedirectingFinder"
]
if SOURCE in sys.path:
    sys.path.remove(SOURCE)
sys.path.insert(0, SOURCE)

for name in list(sys.modules):
    if name == "tensorstudio" or name.startswith("tensorstudio."):
        del sys.modules[name]
