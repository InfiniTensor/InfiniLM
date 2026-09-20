#include "fp8_blockwise.hpp"

#include "gptq_marlin.hpp"
#include "marlin_support.hpp"
#include "marlin_utils.hpp"

#include <infinicore/ops/add.hpp>
#include <infinicore/ops/fp8_blockwise_dequantize.hpp>
#include <infinicore/ops/fp8_blockwise_gemm.hpp>
#include <infinicore/ops/linear.hpp>

#include <cstring>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

namespace {

// float -> IEEE half bits, round-to-nearest-even (scale values are finite and
// non-negative, but handle subnormals/overflow for robustness).
uint16_t f32_to_f16_bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000u;
    const int exp = static_cast<int>((u >> 23) & 0xFFu) - 127 + 15;
    const uint32_t mantissa = u & 0x7FFFFFu;
    if (exp <= 0) {
        if (exp < -10) {
            return static_cast<uint16_t>(sign); // underflow to zero
        }
        const uint32_t m = (mantissa | 0x800000u) >> (14 - exp);
        return static_cast<uint16_t>(sign | ((m + 1) >> 1));
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7C00u); // overflow to inf
    }
    const uint32_t rounding = 0xFFFu + ((mantissa >> 13) & 1u);
    return static_cast<uint16_t>(sign | ((static_cast<uint32_t>(exp) << 10) + ((mantissa + rounding) >> 13)));
}

} // namespace

FP8Blockwise::FP8Blockwise(const nlohmann::json &quant_config)
    : BaseQuantization(quant_config) {
    auto block_size = get_or<std::vector<size_t>>("weight_block_size", {128, 128});
    if (block_size.size() != 2 || block_size[0] == 0 || block_size[1] == 0) {
        throw std::runtime_error("FP8Blockwise: weight_block_size must be [BM, BN] with positive entries");
    }
    block_m_ = block_size[0];
    block_n_ = block_size[1];
}

std::vector<ParamDescriptor> FP8Blockwise::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {
    if (in_features % block_n_ != 0 || out_features % block_m_ != 0) {
        throw std::runtime_error("FP8Blockwise: in_features (" + std::to_string(in_features) + ") and out_features (" + std::to_string(out_features) + ") must be divisible by weight_block_size [" + std::to_string(block_m_) + ", " + std::to_string(block_n_) + "]");
    }
    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, infinicore::DataType::F8, split_dim, tp_rank, tp_size});
    // Scale is split across TP ranks along the same dim as the weight; the
    // per-rank shard size follows from the weight shard divided by the block.
    descs.push_back({"weight_scale_inv", {out_features / block_m_, in_features / block_n_}, infinicore::DataType::F32, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim >= 0 ? 0 : -1, split_dim >= 0 ? tp_rank : 0, split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor FP8Blockwise::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();

    // Fused path (decode): the fp8_blockwise_gemm kernel reads every FP8
    // weight byte exactly once and dequantizes in registers, avoiding the
    // materialized BF16 weight of the naive path. The operator dispatches
    // internally: SIMT warp-per-row for M <= 8 (measured on RTX 5090,
    // Qwen3-8B-FP8: bs=1 141 vs 28 tok/s, bs=8 412 vs 218 against naive),
    // a tensor-core mma.m16n8k16 kernel for 9 <= M <= 32 with F16/BF16
    // activations (the SIMT M_TILE kernels go instruction-throughput bound
    // there; INFINIOP_FP8_GEMM_MMA=0 forces SIMT for A/B). M > 32 (prefill)
    // stays on the naive cuBLAS path, as does any M with F32 activations
    // (SIMT M_TILE fallback inside the operator). The fused kernel
    // implements alpha == 1 only. Set INFINILM_FP8_FUSED_GEMM=0 to force
    // the naive path (A/B testing).
    static const bool fused_gemm_enabled = [] {
        const char *env = std::getenv("INFINILM_FP8_FUSED_GEMM");
        return env == nullptr || env[0] != '0';
    }();
    const auto &weight = params.at("weight");
    const size_t k = weight->size(1);
    const size_t m = input_contiguous->numel() / k;
    if (fused_gemm_enabled && alpha == 1.0f && m <= 32
        && input->device().getType() == infinicore::Device::Type::NVIDIA
        && block_m_ % 16 == 0 && block_n_ % 128 == 0 && k % 128 == 0) {
        auto flat_input = input_contiguous->view({m, k});
        auto output = infinicore::op::fp8_blockwise_gemm(
            flat_input, weight, params.at("weight_scale_inv"));
        if (has_bias) {
            auto bias = params.at("bias");
            infinicore::op::add_(output, output, bias->as_strided(output->shape(), {0, 1}));
        }
        auto out_shape = input_contiguous->shape();
        out_shape.back() = output->size(1);
        return output->view(out_shape);
    }

    // Naive path (prefill / fallback): dequantize the block-scaled FP8 weight
    // to the activation dtype, then run the standard GEMM.
    auto dequant_weight = infinicore::op::fp8_blockwise_dequantize(
        weight, params.at("weight_scale_inv"), input->dtype());

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }
    return infinicore::op::linear(input_contiguous, dequant_weight, bias_opt, alpha);
}

std::vector<SplitParam> FP8Blockwise::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {
    std::vector<SplitParam> result;
    const auto &weight = params.at("weight");
    const auto &weight_scale_inv = params.at("weight_scale_inv");
    const auto bias_it = params.find("bias");

    // SplitInfo start/size are in units of the weight's narrow_dim extent.
    // weight_scale_inv holds one entry per block, so its narrow range along
    // the same dim is divided by the block size (BM for dim0, BN for dim1).
    const size_t block = (narrow_dim == 0) ? block_m_ : block_n_;

    for (const auto &split : splits) {
        if (split.start % block != 0 || split.size % block != 0) {
            throw std::runtime_error("FP8Blockwise: split range at start=" + std::to_string(split.start) + " size=" + std::to_string(split.size) + " is not aligned to block size " + std::to_string(block));
        }
        result.push_back({split.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight->narrow({{static_cast<size_t>(narrow_dim), split.start, split.size}}),
                              narrow_dim, tp_rank, tp_size, split.num_shards)});
        result.push_back({split.prefix + ".weight_scale_inv",
                          infinicore::nn::Parameter(
                              weight_scale_inv->narrow({{static_cast<size_t>(narrow_dim), split.start / block, split.size / block}}),
                              narrow_dim, tp_rank, tp_size, split.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({split.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, split.start, split.size}}),
                                  0, tp_rank, tp_size, split.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> FP8Blockwise::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int) const {
    const auto weight_it = params.find("weight");
    const auto scale_it = params.find("weight_scale_inv");
    if (weight_it == params.end() || scale_it == params.end()) {
        throw std::runtime_error(
            "FP8Blockwise: post-load processing requires weight and weight_scale_inv");
    }
#if INFINILM_ENABLE_MARLIN
    // Fused path (opt-in via INFINILM_FP8_MARLIN=1): convert the block-scaled
    // FP8 weight to Marlin's 8-bit layout so forward runs the fused
    // dequantize-in-GEMM kernel instead of materializing a BF16 weight per
    // step. DeepSeek/Qwen block scales [N/BM, K/BN] are expanded (exact
    // repetition) to Marlin's per-K-group, per-column grid [K/128, N]; only
    // the canonical 128x128 block maps onto Marlin's group_size=128
    // (group_blocks=8) FP8 instantiation.
    //
    // NOTE: the InfiniCore Marlin GEMM operator requires TVM-FFI headers at
    // build time (ENABLE_TVM_API; otherwise calculate() is a silent no-op),
    // and its kernels are currently broken on sm_120 (deadlock on CUDA 12.8,
    // garbage output on CUDA 13.2 — see dev_fp8/marlin_sm120_issue.md). The
    // naive dequantize+GEMM path therefore stays the default.
    static const bool fp8_marlin_enabled = [] {
        const char *env = std::getenv("INFINILM_FP8_MARLIN");
        return env != nullptr && env[0] == '1';
    }();
    if (fp8_marlin_enabled && device.getType() == infinicore::Device::Type::NVIDIA && block_m_ == 128 && block_n_ == 128) {
        const auto &weight = weight_it->second;   // [N, K] F8
        const auto &scales = scale_it->second;    // [N/128, K/128] F32
        const size_t size_n = weight->size(0);
        const size_t size_k = weight->size(1);
        const size_t num_groups = size_k / 128;
        if (marlin::supports_shape(size_k, size_n, 128)) {
            // 1) Weight: [N, K] FP8 bytes -> GPTQ-style qweight [K/4, N] I32
            //    (4 consecutive K bytes per word, little-endian; gptq_value()
            //    extracts byte (k%4) from bits 8*(k%4)).
            auto weight_cpu = weight->contiguous()->to(infinicore::Device::cpu());
            const auto *w_bytes = reinterpret_cast<const uint8_t *>(weight_cpu->data());
            std::vector<int32_t> packed(size_k / 4 * size_n, 0);
            for (size_t n = 0; n < size_n; ++n) {
                const uint8_t *row = w_bytes + n * size_k;
                uint32_t *dst_col = reinterpret_cast<uint32_t *>(packed.data()) + n;
                for (size_t kp = 0; kp < size_k / 4; ++kp) {
                    dst_col[kp * size_n] = static_cast<uint32_t>(row[4 * kp]) | (static_cast<uint32_t>(row[4 * kp + 1]) << 8) | (static_cast<uint32_t>(row[4 * kp + 2]) << 16) | (static_cast<uint32_t>(row[4 * kp + 3]) << 24);
                }
            }
            auto qweight_cpu = marlin::make_i32_tensor(packed, {size_k / 4, size_n}, infinicore::Device::cpu());
            auto perm_empty_cpu = marlin::make_empty_i32(infinicore::Device::cpu());
            // CPU input keeps the repack on the host; result is moved to the device.
            params["qweight"] = marlin::gptq_marlin_repack(qweight_cpu, perm_empty_cpu, size_k, size_n, 8)->to(device);

            // 2) Scales: [N/128, K/128] F32 -> [K/128, N] with each block value
            //    repeated across the 128 output columns of its block (exact),
            //    cast to the activation dtype expected by the kernel (the bias,
            //    when present, is stored in the model dtype).
            auto scales_cpu = scales->contiguous()->to(infinicore::Device::cpu());
            const auto *s_src = reinterpret_cast<const float *>(scales_cpu->data());
            const auto bias_it = params.find("bias");
            const auto act_dtype = (bias_it != params.end()) ? bias_it->second->dtype() : infinicore::DataType::BF16;
            if (act_dtype != infinicore::DataType::BF16 && act_dtype != infinicore::DataType::F16) {
                return nullptr;
            }
            auto scales_exp = infinicore::Tensor::empty({num_groups, size_n}, act_dtype, infinicore::Device::cpu());
            auto *s_dst = reinterpret_cast<uint16_t *>(scales_exp->data());
            for (size_t g = 0; g < num_groups; ++g) {
                for (size_t n = 0; n < size_n; ++n) {
                    const float s = s_src[(n / 128) * num_groups + g];
                    uint32_t u;
                    std::memcpy(&u, &s, sizeof(u));
                    if (act_dtype == infinicore::DataType::BF16) {
                        const uint32_t rounding = 0x7FFFu + ((u >> 16) & 1u);
                        s_dst[g * size_n + n] = static_cast<uint16_t>((u + rounding) >> 16);
                    } else {
                        s_dst[g * size_n + n] = f32_to_f16_bits(s);
                    }
                }
            }
            params["scales"] = marlin::permute_scales(scales_exp, size_k, size_n, 128)->to(device);

            params["qzeros"] = marlin::make_empty_i32(device);
            params["g_idx"] = marlin::make_empty_i32(device);
            params["perm"] = marlin::make_empty_i32(device);
            params["global_scales"] = marlin::make_empty_i32(device);
            params.erase("weight");
            params.erase("weight_scale_inv");

            return std::make_shared<GPTQMarlin>(
                get_config(), size_k, size_n, /*is_k_full=*/true, marlin::FE4M3FN_ID);
        }
    }
#endif
    return nullptr;
}

} // namespace infinilm::quantization
