"""Heartbeat — a tiny systolic array simulator."""
from .core import Result, Snapshot, matmul_ref, allclose
from . import output_stationary, weight_stationary

DATAFLOWS = {
    "os": output_stationary.simulate,
    "ws": weight_stationary.simulate,
}


def simulate(A, B, dataflow: str = "os", rows=None, cols=None, trace: bool = False) -> Result:
    """Run C = A @ B on a systolic array. dataflow is 'os' or 'ws'."""
    try:
        fn = DATAFLOWS[dataflow]
    except KeyError:
        raise ValueError(f"unknown dataflow {dataflow!r}; choose from {sorted(DATAFLOWS)}") from None
    return fn(A, B, rows=rows, cols=cols, trace=trace)


__all__ = ["simulate", "Result", "Snapshot", "matmul_ref", "allclose",
           "output_stationary", "weight_stationary", "DATAFLOWS"]
__version__ = "0.1.0"
