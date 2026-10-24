#include "heartbeat.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace heartbeat {

Matrix matmul_reference(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) throw std::invalid_argument("inner dimensions differ");
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i)
        for (int t = 0; t < A.cols; ++t) {
            const double a = A(i, t);
            for (int j = 0; j < B.cols; ++j) C(i, j) += a * B(t, j);
        }
    return C;
}

Stats simulate(Dataflow df, const Matrix& A, const Matrix& B, int rows, int cols, Matrix& C) {
    switch (df) {
        case Dataflow::OutputStationary: return simulate_output_stationary(A, B, rows, cols, C);
        case Dataflow::WeightStationary: return simulate_weight_stationary(A, B, rows, cols, C);
    }
    throw std::invalid_argument("unknown dataflow");
}

bool allclose(const Matrix& X, const Matrix& Y, double tol) {
    if (X.rows != Y.rows || X.cols != Y.cols) return false;
    for (std::size_t i = 0; i < X.data.size(); ++i) {
        const double a = X.data[i], b = Y.data[i];
        if (std::fabs(a - b) > tol * std::max({1.0, std::fabs(a), std::fabs(b)})) return false;
    }
    return true;
}

}  // namespace heartbeat
