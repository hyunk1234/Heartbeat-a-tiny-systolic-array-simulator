// Weight-stationary array: PE(t,j) holds B[t][j] (loaded one row per cycle,
// K cycles per tile). A[i][t] enters row t at cycle i + t and shifts right;
// partial sums start at 0 above row 0 and shift down, so C[i][j] leaves the
// bottom of column j at cycle i + (K-1) + j. A tile takes m + K + N - 2
// compute cycles. k larger than the array is split and summed in an
// accumulator buffer below the array.
#include "heartbeat.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace heartbeat {
namespace {

struct Reg {
    double v = 0.0;
    int tag = -1;  // which row of A this value belongs to; -1 = bubble
};

void run_tile(const Matrix& A, const Matrix& B, int t0, int j0, int K, int N, Matrix& C, Stats& s) {
    const int m = A.rows;
    const std::size_t n = static_cast<std::size_t>(K) * N;
    std::vector<double> W(n);
    for (int t = 0; t < K; ++t)
        for (int j = 0; j < N; ++j) W[static_cast<std::size_t>(t) * N + j] = B(t0 + t, j0 + j);

    std::vector<Reg> x(n), ps(n), nx(n), nps(n);
    const long long T = static_cast<long long>(m) + K + N - 2;

    for (long long cyc = 0; cyc < T; ++cyc) {
        for (int t = 0; t < K; ++t) {
            for (int j = 0; j < N; ++j) {
                const std::size_t p = static_cast<std::size_t>(t) * N + j;
                Reg a;
                if (j == 0) {
                    const long long i = cyc - t;
                    if (i >= 0 && i < m) a = {A(static_cast<int>(i), t0 + t), static_cast<int>(i)};
                } else {
                    a = x[p - 1];
                }
                Reg out;
                if (a.tag >= 0) {
                    double pin = 0.0;
                    if (t > 0) {
                        const Reg& above = ps[p - N];
                        if (above.tag != a.tag) throw std::logic_error("partial sum and activation out of step");
                        pin = above.v;
                    }
                    out = {pin + W[p] * a.v, a.tag};
                    ++s.macs;
                    if (t == K - 1) C(a.tag, j0 + j) += out.v;  // leaves the array into the accumulator
                }
                nx[p] = a;
                nps[p] = out;
            }
        }
        std::swap(x, nx);
        std::swap(ps, nps);
    }
    s.cycles += T;
    s.load_cycles += K;
    ++s.tiles;
}

}  // namespace

Stats simulate_weight_stationary(const Matrix& A, const Matrix& B, int rows, int cols, Matrix& C) {
    if (A.rows < 1 || A.cols < 1 || B.cols < 1) throw std::invalid_argument("matrices must be non-empty");
    if (A.cols != B.rows) throw std::invalid_argument("inner dimensions differ");
    const int k = A.cols, n = B.cols;
    if (rows <= 0) rows = k;
    if (cols <= 0) cols = n;

    C = Matrix(A.rows, n);
    Stats s;
    s.rows = rows;
    s.cols = cols;
    for (int j0 = 0; j0 < n; j0 += cols)
        for (int t0 = 0; t0 < k; t0 += rows)
            run_tile(A, B, t0, j0, std::min(rows, k - t0), std::min(cols, n - j0), C, s);
    return s;
}

}  // namespace heartbeat
