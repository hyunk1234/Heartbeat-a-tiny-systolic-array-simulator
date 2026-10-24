# Heartbeat — a tiny systolic array simulator

A cycle-accurate simulator of a 2-D systolic array doing matrix
multiplication, with a **C++17 core**, a **C99 API**, and a **Python** front end
(reference implementation, ASCII visualizer, CLI) that calls the C++ core
through `ctypes`. No third-party dependencies. Data pulses through a grid of
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

## C / C++ core

```bash
make            # builds build/libheartbeat.{so,dylib}, hb_tests, hb_bench, c_example
make test       # C++ unit tests + C example + Python tests (incl. C++ vs Python cross-check)
make bench      # 128x128x128 on a 32x32 array
python3 -m heartbeat bench                       # Python vs C++ wall-clock, same input
python3 -m heartbeat compare --backend native    # any command can use the C++ core
```

- `cpp/include/heartbeat.hpp` — C++ API (`heartbeat::simulate`, `Matrix`, `Stats`)
- `cpp/include/heartbeat.h` — `extern "C"` API (`hb_matmul`, `hb_last_error`), exceptions
  are caught at the boundary and turned into error codes
- `examples/c_example.c` — calling the simulator from plain C
- `heartbeat/native.py` — `ctypes` binding; the Python tests require the C++ core to match
  the Python model on results **and** on cycle, load, MAC and tile counts

The C++ core simulates ~600 M PE-cycles/s and runs about 50–75× faster than the Python
model end to end (including ctypes marshaling).
Building on macOS needs the Xcode command line tools (`xcode-select --install`).

## Use from Python

```python
from heartbeat import simulate
res = simulate(A, B, dataflow="ws", rows=4, cols=4, trace=True)
res.C, res.total_cycles, res.utilization, res.trace[0].active
fast = simulate(A, B, dataflow="ws", rows=4, cols=4, backend="native")   # C++ core
```

## Layout

```
heartbeat/core.py               Result/Snapshot types, reference matmul
heartbeat/output_stationary.py  OS array
heartbeat/weight_stationary.py  WS array (+ k-tiling with an accumulator buffer)
heartbeat/visualize.py          ASCII frames
heartbeat/cli.py                run / compare commands
heartbeat/native.py             ctypes binding to the C++ core
tests/test_heartbeat.py         correctness, cycle counts, wavefront timing, CLI, C++ parity
cpp/include/                    heartbeat.hpp (C++ API), heartbeat.h (C API)
cpp/src/                        OS + WS arrays in C++, C wrapper
cpp/tests/test_main.cpp         C++ unit tests (3000+ checks)
cpp/tools/hb_bench.cpp          C++ benchmark
examples/c_example.c            C usage example
Makefile                        build / test / bench (Linux + macOS)
```
