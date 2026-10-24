// Output-stationary array: PE(i,j) accumulates C[i][j]. Row i of A enters
// from the left delayed i cycles, column j of B from the top delayed j
// cycles; A shifts right and B shifts down one PE per clock.
// A tile of r x c PEs with inner dimension k takes k + r + c - 2 cycles.
#include "heartbeat.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace heartbeat {
namespace {

struct Reg {
    double v = 0.0;
    bool valid = false;
};

void run_tile(const Matrix& A, const Matrix& B, int i0, int j0, int r, int c, Matrix& C, Stats& s) {
    const int k = A.cols;
    const std::size_t n = static_cast<std::size_t>(r) * c;
    std::vector<Reg> x(n), y(n), nx(n), ny(n);
    std::vector<double> acc(n, 0.0);
    const long long T = static_cast<long long>(k) + r + c - 2;

    for (long long t = 0; t < T; ++t) {
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                const std::size_t p = static_cast<std::size_t>(i) * c + j;
                Reg a, b;
                if (j == 0) {
                    const long long sidx = t - i;
                    if (sidx >= 0 && sidx < k) a = {A(i0 + i, static_cast<int>(sidx)), true};
                } else {
                    a = x[p - 1];
                }
                if (i == 0) {
                    const long long sidx = t - j;
                    if (sidx >= 0 && sidx < k) b = {B(static_cast<int>(sidx), j0 + j), true};
                } else {
                    b = y[p - c];
                }
                if (a.valid && b.valid) {
                    acc[p] += a.v * b.v;
                    ++s.macs;
                }
                nx[p] = a;
                ny[p] = b;
            }
        }
        std::swap(x, nx);  // every register latches on the same clock edge
        std::swap(y, ny);
    }
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) C(i0 + i, j0 + j) = acc[static_cast<std::size_t>(i) * c + j];
    s.cycles += T;
    ++s.tiles;
}

}  // namespace

Stats simulate_output_stationary(const Matrix& A, const Matrix& B, int rows, int cols, Matrix& C) {
    if (A.rows < 1 || A.cols < 1 || B.cols < 1) throw std::invalid_argument("matrices must be non-empty");
    if (A.cols != B.rows) throw std::invalid_argument("inner dimensions differ");
    const int m = A.rows, n = B.cols;
    if (rows <= 0) rows = m;
    if (cols <= 0) cols = n;

    C = Matrix(m, n);
    Stats s;
    s.rows = rows;
    s.cols = cols;
    for (int i0 = 0; i0 < m; i0 += rows)
        for (int j0 = 0; j0 < n; j0 += cols)
            run_tile(A, B, i0, j0, std::min(rows, m - i0), std::min(cols, n - j0), C, s);
    return s;
}

}  // namespace heartbeat
