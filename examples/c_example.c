/* Calling the Heartbeat simulator from plain C99 through the C API. */
#include <stdio.h>

#include "heartbeat.h"

static void print_matrix(const char* name, const double* M, int r, int c) {
    printf("%s =\n", name);
    for (int i = 0; i < r; ++i) {
        printf("  [");
        for (int j = 0; j < c; ++j) printf(" %6.1f", M[i * c + j]);
        printf(" ]\n");
    }
}

int main(void) {
    enum { M = 3, K = 4, N = 2 };
    const double A[M * K] = {1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12};
    const double B[K * N] = {1, 0,
                             0, 1,
                             1, 0,
                             0, 1};
    double C[M * N];
    const char* names[] = {"output-stationary", "weight-stationary"};

    printf("Heartbeat %s (C API)\n", hb_version());
    print_matrix("A", A, M, K);
    print_matrix("B", B, K, N);

    for (int df = HB_OUTPUT_STATIONARY; df <= HB_WEIGHT_STATIONARY; ++df) {
        hb_stats s;
        /* 2x2 physical array: forces tiling */
        if (hb_matmul(df, A, B, M, K, N, 2, 2, C, &s) != HB_OK) {
            fprintf(stderr, "error: %s\n", hb_last_error());
            return 1;
        }
        printf("\n%s on a %dx%d array: %lld tiles, %lld cycles (+%lld load), %lld MACs\n",
               names[df], s.rows, s.cols, s.tiles, s.cycles, s.load_cycles, s.macs);
        print_matrix("C", C, M, N);
    }
    return 0;
}
