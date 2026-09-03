// Device-side driver for ggml_blocks.h, used by scripts/gguf_routeb_blocks_ref.py.
//
//   nvcc -O2 -std=c++17 -I <InfiniCore>/src/infiniop/ops/linear_gguf \
//       gguf_routeb_blocks_probe.cu -o blocks_probe_cuda
//
//   blocks_probe_cuda <ggml_type> <n_blocks> <in.bin> <out_f32.bin> <out_bf16.bin>
//
// Same job as gguf_routeb_blocks_probe.cpp, but every block is decoded by one
// thread through the very same ggml_blocks.h entry points, which is what proves
// the header is device-safe (no host-only call, no unaligned struct punning) and
// that the host and device results are bit-identical.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml_blocks.h"

__global__ void decode_f32_kernel(int32_t type, const uint8_t *blk, int64_t n_blocks,
                                  int32_t bytes, int32_t elems, float *out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) return;
    ggml_blocks::decode_blocks(type, blk + (int64_t)i * bytes, 1, out + i * elems);
}

__global__ void decode_bf16_kernel(int32_t type, const uint8_t *blk, int64_t n_blocks,
                                   int32_t bytes, int32_t elems, uint16_t *out) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks) return;
    ggml_blocks::decode_blocks_bf16<ggml_blocks::QK_K>(type, blk + (int64_t)i * bytes, 1,
                                                       out + i * elems);
}

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            std::fprintf(stderr, "probe cuda: %s failed: %s\n", #call,               \
                         cudaGetErrorString(err__));                                  \
            return 5;                                                                 \
        }                                                                             \
    } while (0)

int main(int argc, char **argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s <ggml_type> <n_blocks> <in.bin> <out_f32.bin> <out_bf16.bin>\n",
                     argv[0]);
        return 2;
    }
    const int32_t type = std::atoi(argv[1]);
    const int64_t n_blocks = std::atoll(argv[2]);
    const int32_t bytes = ggml_blocks::block_bytes(type);
    const int32_t elems = ggml_blocks::block_elems(type);
    if (bytes < 0 || elems < 0) {
        std::fprintf(stderr, "probe cuda: ggml type %d has no decoder here\n", type);
        return 3;
    }
    if (n_blocks <= 0) {
        std::fprintf(stderr, "probe cuda: n_blocks must be positive\n");
        return 2;
    }

    FILE *in = std::fopen(argv[3], "rb");
    if (!in) {
        std::fprintf(stderr, "probe cuda: cannot open %s\n", argv[3]);
        return 4;
    }
    const size_t want = (size_t)n_blocks * bytes;
    std::vector<uint8_t> buf(want);
    const size_t got = std::fread(buf.data(), 1, want, in);
    std::fclose(in);
    if (got != want) {
        std::fprintf(stderr, "probe cuda: short read on %s (wanted %zu, got %zu)\n", argv[3],
                     want, got);
        return 4;
    }

    uint8_t *d_blk = nullptr;
    float *d_f32 = nullptr;
    uint16_t *d_bf16 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blk, want));
    CUDA_CHECK(cudaMalloc(&d_f32, (size_t)n_blocks * elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bf16, (size_t)n_blocks * elems * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemcpy(d_blk, buf.data(), want, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int64_t blocks_grid = (n_blocks + threads - 1) / threads;
    decode_f32_kernel<<<(unsigned)blocks_grid, threads>>>(type, d_blk, n_blocks, bytes, elems,
                                                          d_f32);
    CUDA_CHECK(cudaGetLastError());
    decode_bf16_kernel<<<(unsigned)blocks_grid, threads>>>(type, d_blk, n_blocks, bytes, elems,
                                                           d_bf16);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_f32((size_t)n_blocks * elems);
    std::vector<uint16_t> h_bf16((size_t)n_blocks * elems);
    CUDA_CHECK(cudaMemcpy(h_f32.data(), d_f32, h_f32.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bf16.data(), d_bf16, h_bf16.size() * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost));
    cudaFree(d_blk);
    cudaFree(d_f32);
    cudaFree(d_bf16);

    FILE *o1 = std::fopen(argv[4], "wb");
    FILE *o2 = std::fopen(argv[5], "wb");
    if (!o1 || !o2) {
        std::fprintf(stderr, "probe cuda: cannot open output files\n");
        return 4;
    }
    const size_t n_f32 = h_f32.size() * sizeof(float);
    const size_t n_bf16 = h_bf16.size() * sizeof(uint16_t);
    const bool ok = std::fwrite(h_f32.data(), 1, n_f32, o1) == n_f32 &&
                    std::fwrite(h_bf16.data(), 1, n_bf16, o2) == n_bf16;
    std::fclose(o1);
    std::fclose(o2);
    if (!ok) {
        std::fprintf(stderr, "probe cuda: short write\n");
        return 4;
    }
    std::printf("probe cuda type=%d n_blocks=%lld elems=%d ok\n", type, (long long)n_blocks,
                elems);
    return 0;
}
