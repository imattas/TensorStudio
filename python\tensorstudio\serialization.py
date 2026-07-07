"""Serialization helpers.

TensorStudio uses pickle for internal object roundtrips. Loading pickle data
from untrusted sources is unsafe because pickle can execute arbitrary code during
deserialization.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .typing import PathLikeStr


def save(obj: Any, path: PathLikeStr) -> None:
    """Serialize a TensorStudio object to a file with pickle."""

    with Path(path).open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: PathLikeStr) -> Any:
    """Load a TensorStudio object from a pickle file.

    Only load files you created or otherwise trust.
    """

    with Path(path).open("rb") as file:
        return pickle.load(file)  # noqa: S301


__all__ = ["load", "save"]
