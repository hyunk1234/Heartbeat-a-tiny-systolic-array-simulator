// Dependency-free unit tests for the C++ core and the C API.
#include "heartbeat.h"
#include "heartbeat.hpp"

#include <cstdio>
#include <random>
#include <stdexcept>

using namespace heartbeat;

static int g_failures = 0, g_checks = 0;
#define CHECK(cond)                                                          \
    do {                                                                     \
        ++g_checks;                                                          \
        if (!(cond)) {                                                       \
            ++g_failures;                                                    \
            std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                    \
    } while (0)

static Matrix random_matrix(int r, int c, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(-5, 5);
    Matrix M(r, c);
    for (auto& v : M.data) v = d(rng);
    return M;
}

static void test_random_shapes() {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dim(1, 9), arr(0, 5);
    for (int iter = 0; iter < 500; ++iter) {
        const int m = dim(rng), k = dim(rng), n = dim(rng);
        const int rows = arr(rng), cols = arr(rng);  // 0 = no tiling
        Matrix A = random_matrix(m, k, rng), B = random_matrix(k, n, rng);
        Matrix ref = matmul_reference(A, B);
        for (Dataflow df : {Dataflow::OutputStationary, Dataflow::WeightStationary}) {
            Matrix C;
            Stats s = simulate(df, A, B, rows, cols, C);
            CHECK(allclose(C, ref));
            CHECK(s.macs == static_cast<long long>(m) * k * n);
            CHECK(s.utilization() <= 1.0);
        }
    }
}

static void test_cycle_counts() {
    std::mt19937 rng(7);
    Matrix A = random_matrix(4, 6, rng), B = random_matrix(6, 5, rng), C;
    Stats os = simulate_output_stationary(A, B, 0, 0, C);
    CHECK(os.cycles == 6 + 4 + 5 - 2);
    CHECK(os.load_cycles == 0);
    Stats ws = simulate_weight_stationary(A, B, 0, 0, C);
    CHECK(ws.cycles == 4 + 6 + 5 - 2);
    CHECK(ws.load_cycles == 6);

    Matrix A2 = random_matrix(4, 3, rng), B2 = random_matrix(3, 4, rng);
    Stats tiled = simulate_output_stationary(A2, B2, 2, 2, C);
    CHECK(tiled.tiles == 4);
    CHECK(tiled.cycles == 4 * (3 + 2 + 2 - 2));
}

static void test_errors() {
    Matrix A(1, 2), B(1, 2), C;
    bool threw = false;
    try {
        simulate_output_stationary(A, B, 0, 0, C);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

static void test_c_api() {
    const double A[] = {1, 2, 3, 4};
    const double B[] = {5, 6, 7, 8};
    double C[4] = {0};
    hb_stats s;
    CHECK(hb_matmul(HB_WEIGHT_STATIONARY, A, B, 2, 2, 2, 0, 0, C, &s) == HB_OK);
    CHECK(C[0] == 19 && C[1] == 22 && C[2] == 43 && C[3] == 50);
    CHECK(s.cycles == 2 + 2 + 2 - 2 && s.load_cycles == 2 && s.macs == 8);
    CHECK(hb_matmul(7, A, B, 2, 2, 2, 0, 0, C, &s) == HB_ERR_ARG);
    CHECK(hb_last_error()[0] != '\0');
    CHECK(hb_matmul(HB_OUTPUT_STATIONARY, nullptr, B, 2, 2, 2, 0, 0, C, nullptr) == HB_ERR_ARG);
}

int main() {
    test_random_shapes();
    test_cycle_counts();
    test_errors();
    test_c_api();
    std::printf("%d/%d checks passed\n", g_checks - g_failures, g_checks);
    return g_failures == 0 ? 0 : 1;
}
