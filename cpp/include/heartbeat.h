/* Heartbeat — plain C interface to the C++ simulator.
 *
 * Usable from C, from Python (ctypes) or any language with a C FFI.
 * Matrices are row-major arrays of double.
 */
#ifndef HEARTBEAT_H
#define HEARTBEAT_H

#if defined(_WIN32)
#  define HB_API __declspec(dllexport)
#else
#  define HB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum { HB_OUTPUT_STATIONARY = 0, HB_WEIGHT_STATIONARY = 1 };
enum { HB_OK = 0, HB_ERR_ARG = -1, HB_ERR_INTERNAL = -2 };

typedef struct {
    long long cycles;
    long long load_cycles;
    long long macs;
    long long tiles;
    int rows;
    int cols;
} hb_stats;

/* C (m x n) = A (m x k) * B (k x n) on a rows x cols array.
 * rows/cols <= 0 pick a single-tile array. stats may be NULL.
 * Returns HB_OK, or a negative error code (see hb_last_error()). */
HB_API int hb_matmul(int dataflow, const double* A, const double* B,
                     int m, int k, int n, int rows, int cols,
                     double* C, hb_stats* stats);

/* Message for the most recent error on this thread ("" if none). */
HB_API const char* hb_last_error(void);

HB_API const char* hb_version(void);

#ifdef __cplusplus
}
#endif

#endif /* HEARTBEAT_H */
