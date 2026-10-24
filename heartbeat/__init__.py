"""Heartbeat — a tiny systolic array simulator (Python reference + C++ core)."""
from .core import Result, Snapshot, matmul_ref, allclose
from . import output_stationary, weight_stationary, native

DATAFLOWS = {
    "os": output_stationary.simulate,
    "ws": weight_stationary.simulate,
}
BACKENDS = ("python", "native")


def simulate(A, B, dataflow: str = "os", rows=None, cols=None, trace: bool = False,
             backend: str = "python") -> Result:
    """Run C = A @ B on a systolic array.

    dataflow: 'os' (output-stationary) or 'ws' (weight-stationary).
    backend:  'python' (supports trace=True) or 'native' (the C++ core, via ctypes).
    """
    if backend == "native":
        if trace:
            raise ValueError("trace=True is only supported by the python backend")
        return native.simulate(A, B, dataflow, rows, cols)
    if backend != "python":
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    try:
        fn = DATAFLOWS[dataflow]
    except KeyError:
        raise ValueError(f"unknown dataflow {dataflow!r}; choose from {sorted(DATAFLOWS)}") from None
    return fn(A, B, rows=rows, cols=cols, trace=trace)


__all__ = ["simulate", "Result", "Snapshot", "matmul_ref", "allclose",
           "output_stationary", "weight_stationary", "native", "DATAFLOWS", "BACKENDS"]
__version__ = "0.2.0"
