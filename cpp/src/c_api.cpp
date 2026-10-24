// extern "C" wrapper so the C++ core can be called from C, Python, etc.
#include "heartbeat.h"
#include "heartbeat.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string g_last_error;

int fail(int code, const char* msg) {
    g_last_error = msg;
    return code;
}
}  // namespace

extern "C" {

int hb_matmul(int dataflow, const double* A, const double* B, int m, int k, int n,
              int rows, int cols, double* C, hb_stats* stats) {
    g_last_error.clear();
    if (!A || !B || !C) return fail(HB_ERR_ARG, "null matrix pointer");
    if (m < 1 || k < 1 || n < 1) return fail(HB_ERR_ARG, "dimensions must be positive");
    if (dataflow != HB_OUTPUT_STATIONARY && dataflow != HB_WEIGHT_STATIONARY)
        return fail(HB_ERR_ARG, "unknown dataflow");
    try {
        using namespace heartbeat;
        Matrix a(m, k), b(k, n), c;
        std::copy(A, A + a.data.size(), a.data.begin());
        std::copy(B, B + b.data.size(), b.data.begin());
        const Stats s = simulate(static_cast<Dataflow>(dataflow), a, b, rows, cols, c);
        std::copy(c.data.begin(), c.data.end(), C);
        if (stats) *stats = {s.cycles, s.load_cycles, s.macs, s.tiles, s.rows, s.cols};
        return HB_OK;
    } catch (const std::invalid_argument& e) {
        return fail(HB_ERR_ARG, e.what());
    } catch (const std::exception& e) {
        return fail(HB_ERR_INTERNAL, e.what());
    }
}

const char* hb_last_error(void) { return g_last_error.c_str(); }

const char* hb_version(void) { return "0.2.0"; }

}  // extern "C"
