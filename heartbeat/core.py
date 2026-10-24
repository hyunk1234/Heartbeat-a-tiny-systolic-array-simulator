"""Shared types and helpers for the Heartbeat simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

Matrix = List[List[float]]


@dataclass
class Snapshot:
    """State of every PE at the end of one clock cycle.

    Output-stationary: ``x`` = A values moving right, ``y`` = B values moving
    down, ``acc`` = stationary accumulators.
    Weight-stationary: ``x`` = activations moving right, ``y`` = stationary
    weights, ``acc`` = partial sums moving down.
    ``None`` means the register holds no valid data (a bubble).
    """

    cycle: int
    x: List[List[Optional[float]]]
    y: List[List[Optional[float]]]
    acc: List[List[Optional[float]]]
    active: List[List[bool]]
    tile: Tuple[int, int] = (0, 0)


@dataclass
class Result:
    C: Matrix
    cycles: int          # compute cycles (first input in -> last result out)
    load_cycles: int     # cycles spent preloading weights (weight-stationary only)
    macs: int            # multiply-accumulates actually performed
    rows: int            # physical array height
    cols: int            # physical array width
    tiles: int = 1
    dataflow: str = ""
    trace: List[Snapshot] = field(default_factory=list)

    @property
    def total_cycles(self) -> int:
        return self.cycles + self.load_cycles

    @property
    def utilization(self) -> float:
        """Fraction of PE-cycles that did a useful MAC."""
        denom = self.rows * self.cols * self.total_cycles
        return self.macs / denom if denom else 0.0


def shape(M: Sequence[Sequence[float]]) -> Tuple[int, int]:
    if not M or not M[0]:
        raise ValueError("matrix must be non-empty")
    w = len(M[0])
    if any(len(r) != w for r in M):
        raise ValueError("matrix rows must all be the same length")
    return len(M), w


def check_matmul(A: Matrix, B: Matrix) -> Tuple[int, int, int]:
    m, k = shape(A)
    k2, n = shape(B)
    if k != k2:
        raise ValueError(f"inner dimensions differ: A is {m}x{k}, B is {k2}x{n}")
    return m, k, n


def grid(rows: int, cols: int, value=None) -> list:
    return [[value for _ in range(cols)] for _ in range(rows)]


def copy_grid(g: list) -> list:
    return [row[:] for row in g]


def matmul_ref(A: Matrix, B: Matrix) -> Matrix:
    """Plain triple-loop reference multiply."""
    m, k, n = check_matmul(A, B)
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(n)] for i in range(m)]


def allclose(X: Matrix, Y: Matrix, tol: float = 1e-9) -> bool:
    return len(X) == len(Y) and all(
        len(rx) == len(ry) and all(abs(a - b) <= tol * max(1.0, abs(a), abs(b)) for a, b in zip(rx, ry))
        for rx, ry in zip(X, Y)
    )
