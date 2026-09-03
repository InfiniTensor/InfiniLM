// Host-side driver for ggml_blocks.h, used by scripts/gguf_routeb_blocks_ref.py.
//
//   g++ -O2 -std=c++17 -I <InfiniCore>/src/infiniop/ops/linear_gguf \
//       gguf_routeb_blocks_probe.cpp -o blocks_probe_host
//
//   blocks_probe_host <ggml_type> <n_blocks> <in.bin> <out_f32.bin> <out_bf16.bin>
//
// This file is a test harness, not part of any library target: it exists so the
// decoders can be checked against numpy / gguf-py block by block before the
// linear_gguf kernels exist.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml_blocks.h"

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
        std::fprintf(stderr, "probe: ggml type %d has no decoder here\n", type);
        return 3;
    }
    if (n_blocks <= 0) {
        std::fprintf(stderr, "probe: n_blocks must be positive\n");
        return 2;
    }

    FILE *in = std::fopen(argv[3], "rb");
    if (!in) {
        std::fprintf(stderr, "probe: cannot open %s\n", argv[3]);
        return 4;
    }
    const size_t want = (size_t)n_blocks * bytes;
    std::vector<uint8_t> buf(want);
    if (std::fread(buf.data(), 1, want, in) != want) {
        std::fprintf(stderr, "probe: short read on %s (wanted %zu)\n", argv[3], want);
        std::fclose(in);
        return 4;
    }
    std::fclose(in);

    std::vector<float> f32((size_t)n_blocks * elems);
    std::vector<uint16_t> bf16((size_t)n_blocks * elems);
    if (!ggml_blocks::decode_blocks(type, buf.data(), n_blocks, f32.data())) {
        std::fprintf(stderr, "probe: decode_blocks failed\n");
        return 3;
    }
    if (!ggml_blocks::decode_blocks_bf16<ggml_blocks::QK_K>(type, buf.data(), n_blocks,
                                                            bf16.data())) {
        std::fprintf(stderr, "probe: decode_blocks_bf16 failed\n");
        return 3;
    }

    FILE *o1 = std::fopen(argv[4], "wb");
    FILE *o2 = std::fopen(argv[5], "wb");
    if (!o1 || !o2) {
        std::fprintf(stderr, "probe: cannot open output files\n");
        return 4;
    }
    const size_t n_f32 = f32.size() * sizeof(float);
    const size_t n_bf16 = bf16.size() * sizeof(uint16_t);
    const bool ok = std::fwrite(f32.data(), 1, n_f32, o1) == n_f32 &&
                    std::fwrite(bf16.data(), 1, n_bf16, o2) == n_bf16;
    std::fclose(o1);
    std::fclose(o2);
    if (!ok) {
        std::fprintf(stderr, "probe: short write\n");
        return 4;
    }
    std::printf("probe host type=%d n_blocks=%lld elems=%d ok\n", type,
                (long long)n_blocks, elems);
    return 0;
}
