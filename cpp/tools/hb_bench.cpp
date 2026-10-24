// hb_bench: run a large random multiply on the simulated array, verify it,
// and report cycle statistics and wall-clock simulation speed.
//
//   hb_bench [--m 64] [--k 64] [--n 64] [--rows 16] [--cols 16] [--dataflow os|ws|both] [--seed 0]
#include "heartbeat.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

using namespace heartbeat;

static void usage() {
    std::puts("usage: hb_bench [--m N] [--k N] [--n N] [--rows N] [--cols N] [--dataflow os|ws|both] [--seed N]");
}

int main(int argc, char** argv) {
    int m = 64, k = 64, n = 64, rows = 16, cols = 16;
    unsigned seed = 0;
    std::string df = "both";
    for (int i = 1; i < argc; ++i) {
        auto next = [&](void) -> const char* {
            if (i + 1 >= argc) { usage(); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--m")) m = std::atoi(next());
        else if (!std::strcmp(argv[i], "--k")) k = std::atoi(next());
        else if (!std::strcmp(argv[i], "--n")) n = std::atoi(next());
        else if (!std::strcmp(argv[i], "--rows")) rows = std::atoi(next());
        else if (!std::strcmp(argv[i], "--cols")) cols = std::atoi(next());
        else if (!std::strcmp(argv[i], "--seed")) seed = static_cast<unsigned>(std::atoi(next()));
        else if (!std::strcmp(argv[i], "--dataflow")) df = next();
        else { usage(); return std::strcmp(argv[i], "--help") ? 2 : 0; }
    }
    if (m < 1 || k < 1 || n < 1) { std::fputs("dimensions must be positive\n", stderr); return 2; }

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    Matrix A(m, k), B(k, n);
    for (auto& v : A.data) v = d(rng);
    for (auto& v : B.data) v = d(rng);
    const Matrix ref = matmul_reference(A, B);

    std::printf("A %dx%d  B %dx%d  array %dx%d\n", m, k, k, n, rows, cols);
    int rc = 0;
    for (const char* name : {"os", "ws"}) {
        if (df != "both" && df != name) continue;
        const Dataflow flow = std::strcmp(name, "os") == 0 ? Dataflow::OutputStationary : Dataflow::WeightStationary;
        Matrix C;
        const auto t0 = std::chrono::steady_clock::now();
        const Stats s = simulate(flow, A, B, rows, cols, C);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = allclose(C, ref);
        rc |= !ok;
        std::printf("  %s: %lld tiles, %lld compute + %lld load = %lld cycles, %lld MACs, util %.1f%%, "
                    "sim %.2f ms (%.1f M PE-cycles/s)  %s\n",
                    name, s.tiles, s.cycles, s.load_cycles, s.total_cycles(), s.macs, 100.0 * s.utilization(),
                    ms, static_cast<double>(s.rows) * s.cols * s.total_cycles() / (ms * 1e3),
                    ok ? "PASS" : "FAIL");
    }
    return rc;
}
