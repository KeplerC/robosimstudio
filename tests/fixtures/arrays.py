# SPDX-License-Identifier: Apache-2.0
"""A stand-in for a Warp array, for a fast test that hands a measure host data.

Thirty test modules each grew their own: ``SimpleNamespace(numpy=lambda: q)``,
``SimpleNamespace(numpy=q.copy)``, a ``_Arr`` class, an ``_Array`` class. They
differ on the one point that matters -- whether what ``numpy()`` returns is the
array or a copy of it -- and a test that writes through a view it was handed
passes against one stand-in and fails against the other. :func:`host` says
which, by name.

Warp's own rule: a CPU array's ``numpy()`` shares its memory, a CUDA array's is
a copy. ``copy=False`` is the CPU behaviour and the default; ``copy=True`` is
the device one, for a test of code that must write back with ``assign``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["HostArray", "host"]


class HostArray:
    """``numpy()``, ``shape``, ``dtype``, ``len()``, ``assign()`` and ``fill_()`` over a numpy array."""

    def __init__(self, values: Any, dtype: Any = None, *, copy: bool = False) -> None:
        self.data = np.array(values, dtype=dtype)
        self.copy = copy
        self.device = "cpu"

    def numpy(self) -> np.ndarray:
        return self.data.copy() if self.copy else self.data

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __len__(self) -> int:
        return len(self.data)

    def assign(self, values: Any) -> None:
        self.data[...] = np.asarray(values, dtype=self.data.dtype)

    def fill_(self, value: Any) -> None:
        self.data.fill(value)


def host(values: Any, dtype: Any = float, *, copy: bool = False) -> HostArray:
    """``values`` as a :class:`HostArray`; ``copy=True`` makes ``numpy()`` hand out copies, as on a GPU."""
    return HostArray(values, dtype, copy=copy)
