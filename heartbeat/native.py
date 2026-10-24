"""ctypes bindings to the C++ core (build it with `make`).

The library is looked up in $HEARTBEAT_LIB, then build/ next to the package.
"""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

from .core import Result, check_matmul

_EXT = {"darwin": "dylib", "win32": "dll"}.get(sys.platform, "so")
_DATAFLOW = {"os": 0, "ws": 1}
_NAMES = {"os": "output-stationary", "ws": "weight-stationary"}


class _Stats(ctypes.Structure):
    _fields_ = [
        ("cycles", ctypes.c_longlong),
        ("load_cycles", ctypes.c_longlong),
        ("macs", ctypes.c_longlong),
        ("tiles", ctypes.c_longlong),
        ("rows", ctypes.c_int),
        ("cols", ctypes.c_int),
    ]


_lib = None
_load_error: Optional[str] = None


def _candidates():
    env = os.environ.get("HEARTBEAT_LIB")
    if env:
        yield Path(env)
    root = Path(__file__).resolve().parent.parent
    yield root / "build" / f"libheartbeat.{_EXT}"


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    for path in _candidates():
        if path.exists():
            lib = ctypes.CDLL(str(path))
            P = ctypes.POINTER(ctypes.c_double)
            lib.hb_matmul.argtypes = [ctypes.c_int, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, P, ctypes.POINTER(_Stats)]
            lib.hb_matmul.restype = ctypes.c_int
            lib.hb_last_error.restype = ctypes.c_char_p
            lib.hb_version.restype = ctypes.c_char_p
            _lib = lib
            return _lib
    _load_error = "native library not found; run `make` in the project root (or set HEARTBEAT_LIB)"
    return None


def available() -> bool:
    return _load() is not None


def version() -> str:
    lib = _load()
    return lib.hb_version().decode() if lib else ""


def simulate(A, B, dataflow: str = "os", rows: Optional[int] = None, cols: Optional[int] = None) -> Result:
    lib = _load()
    if lib is None:
        raise RuntimeError(_load_error)
    if dataflow not in _DATAFLOW:
        raise ValueError(f"unknown dataflow {dataflow!r}; choose from {sorted(_DATAFLOW)}")
    m, k, n = check_matmul(A, B)
    Arr = ctypes.c_double * (m * k)
    Brr = ctypes.c_double * (k * n)
    a = Arr(*(float(v) for row in A for v in row))
    b = Brr(*(float(v) for row in B for v in row))
    c = (ctypes.c_double * (m * n))()
    st = _Stats()
    rc = lib.hb_matmul(_DATAFLOW[dataflow], a, b, m, k, n, rows or 0, cols or 0, c, ctypes.byref(st))
    if rc != 0:
        msg = lib.hb_last_error().decode()
        raise ValueError(msg) if rc == -1 else RuntimeError(msg)
    C = [[c[i * n + j] for j in range(n)] for i in range(m)]
    return Result(C, st.cycles, st.load_cycles, st.macs, st.rows, st.cols, st.tiles, _NAMES[dataflow])
