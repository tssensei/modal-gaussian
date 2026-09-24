"""Small typed boundaries around NumPy artifact I/O."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from typing import Any, cast

import numpy as np


def save_named_arrays(
    path: str | PathLike[str], arrays: Mapping[str, np.ndarray]
) -> None:
    """Write named NPZ arrays without exposing NumPy's reserved keywords."""

    if "allow_pickle" in arrays:
        raise ValueError("'allow_pickle' is reserved by numpy.savez_compressed")
    writer = cast(Any, np.savez_compressed)
    writer(path, **arrays)


__all__ = ["save_named_arrays"]
