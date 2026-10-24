# Heartbeat — a tiny systolic array simulator

A cycle-accurate, pure-Python (no dependencies) simulator of a 2-D systolic
array doing matrix multiplication. Data pulses through a grid of
multiply-accumulate PEs one hop per clock, like a heartbeat.

Two dataflows:

| Dataflow | What stays put | What moves | Compute cycles per tile |
|---|---|---|---|
| `os` output-stationary | C[i][j] accumulates in PE(i,j) | A → right, B ↓ down | k + r + c − 2 |
| `ws` weight-stationary | B[t][j] preloaded in PE(t,j) | A → right, partial sums ↓ down | m + K + N − 2 (+ K load) |

Matrices bigger than the array are tiled automatically (`--rows/--cols`).
Every result is checked against a plain triple-loop multiply.

## Run

```bash
python3 -m heartbeat run --a "1,2;3,4" --b "5,6;7,8" --trace   # every cycle, printed
python3 -m heartbeat run --dataflow ws --m 3 --k 4 --n 3 --animate  # redrawn in place
python3 -m heartbeat compare --m 8 --k 16 --n 8 --rows 4 --cols 4  # OS vs WS cycles/utilization
python3 -m unittest discover -s tests                          # tests
```

`pip install -e .` also gives you a `heartbeat` command.

In a trace, each box is a PE. A `=` border means that PE did a MAC this cycle.
`·` is an empty register (a bubble). Labels: `A→`/`x→` value moving right,
`B↓` value moving down, `w` stationary weight, `Σ` accumulator, `Σ↓` partial sum moving down.

## Use from Python

```python
from heartbeat import simulate
res = simulate(A, B, dataflow="ws", rows=4, cols=4, trace=True)
res.C, res.total_cycles, res.utilization, res.trace[0].active
```

## Layout

```
heartbeat/core.py               Result/Snapshot types, reference matmul
heartbeat/output_stationary.py  OS array
heartbeat/weight_stationary.py  WS array (+ k-tiling with an accumulator buffer)
heartbeat/visualize.py          ASCII frames
heartbeat/cli.py                run / compare commands
tests/test_heartbeat.py         correctness, cycle counts, wavefront timing, CLI
```
