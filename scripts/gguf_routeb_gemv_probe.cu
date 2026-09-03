// Standalone driver for linear_gguf's two NVIDIA paths, used by
// scripts/gguf_routeb_gemv_check.py:
//
//   M <= kMaxDecodeM  -> launch_gemv_decode   (stage 3.2, decode path)
//   M >  kMaxDecodeM  -> launch_prefill       (stage 3.3, prefill path)
//
// The routing predicate is the one the op itself applies in
// linear_gguf_nvidia.cu::calculate, so a case run here goes through the same
// function the shipped kernel goes through.
//
//   nvcc -O2 -std=c++17 -I <InfiniCore>/src/infiniop/ops/linear_gguf/nvidia \
//       gguf_routeb_gemv_probe.cu -o gemv_probe -lcublas
//
//   gemv_probe <ggml_type> <M> <N> <K> <row_bytes> <a_bf16.bin> <w.bin> <c_bf16.bin>
//
// A test harness, not a library target: it includes the two kernel headers and
// links cublas directly, so the paths can be checked numerically without a
// registered op or an InfiniCore handle.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "linear_gguf_dequant.cuh"

#define CUDA_CHECK(call)                                               \
    do {                                                               \
        cudaError_t err__ = (call);                                    \
        if (err__ != cudaSuccess) {                                    \
            std::fprintf(stderr, "gemv probe: %s failed: %s\n", #call, \
                         cudaGetErrorString(err__));                   \
            return 5;                                                  \
        }                                                              \
    } while (0)

static std::vector<uint8_t> read_all(const char *path, size_t want) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "gemv probe: cannot open %s\n", path);
        exit(4);
    }
    std::vector<uint8_t> buf(want);
    const size_t got = std::fread(buf.data(), 1, want, f);
    std::fclose(f);
    if (got != want) {
        std::fprintf(stderr, "gemv probe: short read on %s (wanted %zu, got %zu)\n", path, want,
                     got);
        exit(4);
    }
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 9) {
        std::fprintf(stderr,
                     "usage: %s <ggml_type> <M> <N> <K> <row_bytes> <a_bf16.bin> <w.bin> "
                     "<c_bf16.bin>\n",
                     argv[0]);
        return 2;
    }
    const int32_t type = std::atoi(argv[1]);
    const int m_count = std::atoi(argv[2]);
    const int n_count = std::atoi(argv[3]);
    const int k = std::atoi(argv[4]);
    const int64_t row_bytes = std::atoll(argv[5]);
    if (m_count <= 0 || n_count <= 0 || k <= 0 || row_bytes <= 0) {
        std::fprintf(stderr, "gemv probe: bad geometry\n");
        return 2;
    }
    const bool prefill = m_count > op::linear_gguf::nvidia::kMaxDecodeM;

    std::vector<uint8_t> h_a = read_all(argv[6], static_cast<size_t>(m_count) * k * 2);
    std::vector<uint8_t> h_w = read_all(argv[7], static_cast<size_t>(n_count) * row_bytes);

    __nv_bfloat16 *d_a = nullptr;
    uint8_t *d_w = nullptr;
    __nv_bfloat16 *d_c = nullptr;
    void *d_scratch = nullptr;
    cublasHandle_t blas = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, h_a.size()));
    CUDA_CHECK(cudaMalloc(&d_w, h_w.size()));
    CUDA_CHECK(cudaMalloc(&d_c, static_cast<size_t>(m_count) * n_count * 2));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), h_a.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), h_w.size(), cudaMemcpyHostToDevice));

    // The prefill scratch is the op's workspace tensor; sized through the same
    // helper the descriptor uses in create(), so the gate also pins that formula.
    const size_t scratch_bytes = op::linear_gguf::nvidia::prefill_scratch_bytes(k);
    if (prefill) {
        CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes));
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "gemv probe: cublasCreate failed\n");
            return 5;
        }
    }

    auto run_once = [&]() -> bool {
        if (prefill) {
            return op::linear_gguf::nvidia::launch_prefill(
                blas, type, d_a, d_w, d_c, m_count, n_count, k, row_bytes,
                d_scratch, scratch_bytes, nullptr);
        }
        return op::linear_gguf::nvidia::launch_gemv_decode(
            type, d_a, d_w, d_c, m_count, n_count, k, row_bytes, nullptr);
    };

    if (!run_once()) {
        std::fprintf(stderr, "gemv probe: %s rejected type %d (no decoder or bad K/row_bytes)\n",
                     prefill ? "prefill" : "gemv", type);
        cublasDestroy(blas);
        return 3;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // One timed run.  Interpret with care: this probe is a numeric harness, the
    // geometry comes from the caller (scripts/gguf_routeb_gemv_check.py) and small
    // N makes the number latency-bound rather than bandwidth-bound.  The bandwidth
    // work is stage 6.
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < 10; ++i) {
        run_once();
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    std::vector<uint8_t> h_c(static_cast<size_t>(m_count) * n_count * 2);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, h_c.size(), cudaMemcpyDeviceToHost));
    cudaFree(d_a);
    cudaFree(d_w);
    cudaFree(d_c);
    cudaFree(d_scratch);
    cublasDestroy(blas);

    FILE *out = std::fopen(argv[8], "wb");
    if (!out) {
        std::fprintf(stderr, "gemv probe: cannot open %s\n", argv[8]);
        return 4;
    }
    const bool wrote = std::fwrite(h_c.data(), 1, h_c.size(), out) == h_c.size();
    std::fclose(out);
    if (!wrote) {
        std::fprintf(stderr, "gemv probe: short write\n");
        return 4;
    }
    std::printf("gemv probe type=%d M=%d N=%d K=%d path=%s ok  %.3f ms/iter  %.2f GiB/s of weight\n",
                type, m_count, n_count, k, prefill ? "prefill" : "gemv", ms / 10.0,
                h_w.size() / (ms / 10.0 * 1e-3) / (1024.0 * 1024.0 * 1024.0));
    return 0;
}
