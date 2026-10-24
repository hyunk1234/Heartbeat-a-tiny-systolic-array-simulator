// Heartbeat — C++ core of the systolic array simulator.
//
// Register-level, cycle-accurate model of a 2-D systolic array computing
// C = A * B. Mirrors the Python reference implementation exactly (same
// cycle counts, same MAC counts) but runs orders of magnitude faster.
#pragma once

#include <cstddef>
#include <vector>

namespace heartbeat {

// Dense row-major matrix of doubles.
struct Matrix {
    int rows = 0;
    int cols = 0;
    std::vector<double> data;

    Matrix() = default;
    Matrix(int r, int c, double fill = 0.0)
        : rows(r), cols(c), data(static_cast<std::size_t>(r) * c, fill) {}

    double& operator()(int i, int j) { return data[static_cast<std::size_t>(i) * cols + j]; }
    double operator()(int i, int j) const { return data[static_cast<std::size_t>(i) * cols + j]; }
};

enum class Dataflow { OutputStationary = 0, WeightStationary = 1 };

struct Stats {
    long long cycles = 0;       // compute cycles, summed over tiles
    long long load_cycles = 0;  // weight preload cycles (weight-stationary only)
    long long macs = 0;         // useful multiply-accumulates
    long long tiles = 0;
    int rows = 0;               // physical array height
    int cols = 0;               // physical array width

    long long total_cycles() const { return cycles + load_cycles; }
    double utilization() const {
        const double denom = static_cast<double>(rows) * cols * total_cycles();
        return denom > 0 ? macs / denom : 0.0;
    }
};

// Plain triple-loop multiply, used to verify the simulator.
Matrix matmul_reference(const Matrix& A, const Matrix& B);

// Run C = A * B on a rows x cols array. rows/cols <= 0 mean "big enough for
// one tile" (m x n for output-stationary, k x n for weight-stationary).
// Throws std::invalid_argument on bad shapes.
Stats simulate_output_stationary(const Matrix& A, const Matrix& B, int rows, int cols, Matrix& C);
Stats simulate_weight_stationary(const Matrix& A, const Matrix& B, int rows, int cols, Matrix& C);
Stats simulate(Dataflow df, const Matrix& A, const Matrix& B, int rows, int cols, Matrix& C);

// Max relative/absolute difference check.
bool allclose(const Matrix& X, const Matrix& Y, double tol = 1e-9);

}  // namespace heartbeat
