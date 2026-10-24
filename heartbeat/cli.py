"""Command line interface: python -m heartbeat {run,compare} ..."""
from __future__ import annotations

import argparse
import random
import sys
import time

from . import simulate, matmul_ref, allclose
from .visualize import render, render_matrix

NAMES = {"os": "output-stationary", "ws": "weight-stationary"}


def parse_matrix(s: str):
    """'1,2;3,4' -> [[1,2],[3,4]]"""
    rows = [r for r in s.split(";") if r.strip()]
    M = [[float(v) for v in r.split(",")] for r in rows]
    return [[int(v) if v.is_integer() else v for v in r] for r in M]


def random_matrix(r, c, rng, floats):
    if floats:
        return [[round(rng.uniform(-1, 1), 3) for _ in range(c)] for _ in range(r)]
    return [[rng.randint(-3, 3) for _ in range(c)] for _ in range(r)]


def _inputs(args):
    rng = random.Random(args.seed)
    A = parse_matrix(args.a) if args.a else random_matrix(args.m, args.k, rng, args.floats)
    B = parse_matrix(args.b) if args.b else random_matrix(len(A[0]), args.n, rng, args.floats)
    return A, B


def summary(res) -> str:
    return (f"{res.dataflow}: {res.rows}x{res.cols} array, {res.tiles} tile(s), "
            f"{res.cycles} compute + {res.load_cycles} load = {res.total_cycles} cycles, "
            f"{res.macs} MACs, utilization {res.utilization:.1%}")


def cmd_run(args) -> int:
    A, B = _inputs(args)
    show = args.trace or args.animate
    res = simulate(A, B, args.dataflow, args.rows, args.cols, trace=show)
    if show:
        for snap in res.trace:
            if args.animate:
                sys.stdout.write("\033[2J\033[H")
            print(render(snap, NAMES[args.dataflow]))
            print()
            if args.animate:
                sys.stdout.flush()
                time.sleep(args.delay)
    print(render_matrix(A, "A"))
    print(render_matrix(B, "B"))
    print(render_matrix(res.C, "C = A·B"))
    ok = allclose(res.C, matmul_ref(A, B))
    print(summary(res))
    print("check vs reference:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_compare(args) -> int:
    A, B = _inputs(args)
    ref = matmul_ref(A, B)
    print(f"A is {len(A)}x{len(A[0])}, B is {len(B)}x{len(B[0])}, array {args.rows or '-'}x{args.cols or '-'}")
    ok = True
    for df in ("os", "ws"):
        res = simulate(A, B, df, args.rows, args.cols)
        good = allclose(res.C, ref)
        ok &= good
        print(" ", summary(res), "" if good else "  <-- MISMATCH")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="heartbeat", description="A tiny systolic array simulator.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def shared(sp):
        sp.add_argument("--m", type=int, default=3, help="rows of A (random input)")
        sp.add_argument("--k", type=int, default=3, help="cols of A / rows of B (random input)")
        sp.add_argument("--n", type=int, default=3, help="cols of B (random input)")
        sp.add_argument("--a", help="explicit A, e.g. '1,2;3,4'")
        sp.add_argument("--b", help="explicit B, e.g. '5,6;7,8'")
        sp.add_argument("--rows", type=int, help="physical array height (enables tiling)")
        sp.add_argument("--cols", type=int, help="physical array width (enables tiling)")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--floats", action="store_true", help="random floats instead of small ints")

    r = sub.add_parser("run", help="simulate one multiply, optionally showing every cycle")
    shared(r)
    r.add_argument("--dataflow", choices=["os", "ws"], default="os")
    r.add_argument("--trace", action="store_true", help="print the array state every cycle")
    r.add_argument("--animate", action="store_true", help="like --trace, redrawn in place")
    r.add_argument("--delay", type=float, default=0.6, help="seconds per frame with --animate")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="cycle counts and utilization for both dataflows")
    shared(c)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
