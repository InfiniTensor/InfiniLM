// Op-level numeric check for the fused rms_norm_rope operator.
//
// Compares, elementwise on GPU:
//   reference: op::rms_norm_ into tmp, then op::rope_ in-place on tmp
//   fused:     op::rms_norm_rope_ in-place on a copy of the same input
// Both paths use identical bf16 inputs / weights / sin-cos tables, so any
// difference comes from the fused kernel's reduction order (fp32 accumulation
// in a smaller thread block) rather than from input rounding.
//
// Build:
//   g++ -std=c++17 -I$HOME/.infini-fa/include dev_perf/op_check_rms_norm_rope.cpp \
//       -L$HOME/.infini-fa/lib -linfinicore_cpp_api -linfiniop -linfinirt -linfiniccl \
//       -Wl,-rpath,$HOME/.infini-fa/lib -o /tmp/op_check_rms_norm_rope
// Run:
//   LD_LIBRARY_PATH=$HOME/.infini-fa/lib:$HOME/.local/cuda-13.2/lib64 /tmp/op_check_rms_norm_rope

#include "infinicore/context/context.hpp"
#include "infinicore/ops/rms_norm.hpp"
#include "infinicore/ops/rms_norm_rope.hpp"
#include "infinicore/ops/rope.hpp"
#include "infinicore/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

static uint16_t f2bf(float f) { // round-to-nearest-even
    uint32_t u;
    std::memcpy(&u, &f, 4);
    uint32_t bias = 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<uint16_t>((u + bias) >> 16);
}
static float bf2f(uint16_t b) {
    uint32_t u = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// deterministic pseudo-random in [-range, range]
static float frand(uint32_t &s, float range) {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>((s >> 8) & 0xFFFF) / 32768.0f - 1.0f) * range;
}

static int run_case(size_t tokens, size_t heads, size_t head_dim,
                    infinicore::nn::RoPE::Algo algo, const char *tag) {
    using namespace infinicore;
    const size_t table_len = 4096, table_dim = head_dim / 2;
    const float eps = 1e-6f;

    // ---- host buffers (bf16) ----
    uint32_t seed = 42;
    std::vector<uint16_t> hx(tokens * heads * head_dim), hw(head_dim);
    for (auto &v : hx) v = f2bf(frand(seed, 2.0f));
    for (auto &v : hw) v = f2bf(0.5f + std::fabs(frand(seed, 1.0f)));
    std::vector<uint16_t> hsin(table_len * table_dim), hcos(table_len * table_dim);
    for (size_t p = 0; p < table_len; ++p)
        for (size_t i = 0; i < table_dim; ++i) {
            float freq = 1.0f / std::pow(10000.0f, 2.0f * i / head_dim);
            hsin[p * table_dim + i] = f2bf(std::sin(p * freq));
            hcos[p * table_dim + i] = f2bf(std::cos(p * freq));
        }
    std::vector<int64_t> hpos(tokens);
    for (size_t t = 0; t < tokens; ++t) hpos[t] = static_cast<int64_t>((t * 37 + 11) % table_len);

    // ---- device tensors ----
    auto dev = context::getDevice();
    auto x_fused = Tensor::empty({tokens, heads, head_dim}, DataType::BF16, dev);
    auto x_ref = Tensor::empty({tokens, heads, head_dim}, DataType::BF16, dev);
    auto tmp = Tensor::empty({tokens, heads, head_dim}, DataType::BF16, dev);
    auto w = Tensor::empty({head_dim}, DataType::BF16, dev);
    auto sin = Tensor::empty({table_len, table_dim}, DataType::BF16, dev);
    auto cos = Tensor::empty({table_len, table_dim}, DataType::BF16, dev);
    auto pos = Tensor::empty({tokens}, DataType::I64, dev);

    context::memcpyH2D(x_fused->data(), hx.data(), hx.size() * 2, false);
    context::memcpyH2D(x_ref->data(), hx.data(), hx.size() * 2, false);
    context::memcpyH2D(w->data(), hw.data(), hw.size() * 2, false);
    context::memcpyH2D(sin->data(), hsin.data(), hsin.size() * 2, false);
    context::memcpyH2D(cos->data(), hcos.data(), hcos.size() * 2, false);
    context::memcpyH2D(pos->data(), hpos.data(), hpos.size() * 8, false);

    // reference chain: rms_norm out-of-place, then rope in-place (engine order)
    op::rms_norm_(tmp, x_ref, w, eps);
    op::rope_(tmp, tmp, pos, sin, cos, algo);
    // fused
    op::rms_norm_rope_(x_fused, w, pos, sin, cos, eps, algo);

    std::vector<uint16_t> out_ref(hx.size()), out_fused(hx.size());
    context::memcpyD2H(out_ref.data(), tmp->data(), hx.size() * 2);
    context::memcpyD2H(out_fused.data(), x_fused->data(), hx.size() * 2);

    size_t n_exact = 0, n_1ulp = 0, n_2ulp = 0, n_worse = 0;
    float max_rel = 0.0f;
    for (size_t i = 0; i < hx.size(); ++i) {
        float a = bf2f(out_ref[i]), b = bf2f(out_fused[i]);
        float diff = std::fabs(a - b);
        float rel = diff / std::max(std::fabs(a), 1e-3f);
        max_rel = std::max(max_rel, rel);
        if (diff == 0.0f) ++n_exact;
        else if (rel <= 0.0078f) ++n_1ulp;   // bf16 ulp ~= 2^-7 relative
        else if (rel <= 0.0156f) ++n_2ulp;
        else ++n_worse;
    }
    std::printf("[%s] tokens=%zu heads=%zu dim=%zu  exact=%.2f%%  1ulp=%zu  2ulp=%zu  worse=%zu  max_rel=%.4f\n",
                tag, tokens, heads, head_dim,
                100.0 * n_exact / hx.size(), n_1ulp, n_2ulp, n_worse, max_rel);
    return n_worse == 0 ? 0 : 1;
}

int main() {
    infinicore::context::setDevice(infinicore::Device(infinicore::Device::Type::NVIDIA, 0));
    int rc = 0;
    // Qwen3-0.6B/1.7B per-layer shape: heads=16(q)/8(k), head_dim=128, GPT_NEOX
    rc |= run_case(97, 16, 128, infinicore::nn::RoPE::Algo::GPT_NEOX, "neox-q");
    rc |= run_case(97, 8, 128, infinicore::nn::RoPE::Algo::GPT_NEOX, "neox-k");
    rc |= run_case(97, 16, 128, infinicore::nn::RoPE::Algo::GPT_J, "gptj-q");
    rc |= run_case(1, 16, 128, infinicore::nn::RoPE::Algo::GPT_NEOX, "neox-single-token");
    std::puts(rc == 0 ? "PASS: all diffs within 2 bf16 ulp" : "FAIL: >2ulp diffs found");
    return rc;
}
