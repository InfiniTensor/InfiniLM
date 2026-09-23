#include "fp8_block.hpp"
#include "marlin_support.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/ops/mul.hpp"
#include <cmath>

#if INFINILM_ENABLE_MARLIN && __has_include("infinicore/ops/awq_marlin_gemm.hpp")
#define INFINILM_ENABLE_FP8_MARLIN 1
#include "infinicore/ops/awq_marlin_gemm.hpp"
#include "marlin_utils.hpp"
#else
#define INFINILM_ENABLE_FP8_MARLIN 0
#endif

namespace infinilm::quantization {

FP8Block::FP8Block(const nlohmann::json &config) : NoneQuantization(config) {
    if (get_or<std::string>("fmt", "") != "e4m3"
        || get_or<std::vector<size_t>>("weight_block_size", {}) != std::vector<size_t>{128, 128}) {
        throw std::runtime_error("FP8 compatibility path requires E4M3 with 128x128 weight blocks.");
    }
}

std::vector<ParamDescriptor> FP8Block::get_param_layout(
    size_t in_features, size_t out_features, int split_dim, int tp_rank,
    int tp_size, int tp_num_heads, const infinicore::DataType &dtype, bool bias) const {
    activation_dtype_ = dtype;
    if (in_features % block_size_ || out_features % block_size_
        || (split_dim == 0 && out_features % (block_size_ * tp_size))
        || (split_dim == 1 && in_features % (block_size_ * tp_size))) {
        throw std::runtime_error("FP8 weight and TP partitions must align to 128-element blocks.");
    }
    auto layout = NoneQuantization::get_param_layout(
        in_features, out_features, split_dim, tp_rank, tp_size, tp_num_heads, dtype, bias);
    layout.front().dtype = infinicore::DataType::F8;
    layout.push_back({"weight_scale_inv", {out_features / block_size_, in_features / block_size_}, infinicore::DataType::F32, split_dim, tp_rank, tp_size});
    return layout;
}

std::vector<SplitParam> FP8Block::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits, int narrow_dim,
    int tp_rank, int tp_size, int tp_num_heads) const {
    auto result = NoneQuantization::split_params(params, splits, narrow_dim, tp_rank, tp_size, tp_num_heads);
    for (const auto &s : splits) {
        if (s.start % block_size_ || s.size % block_size_) {
            throw std::runtime_error("FP8 fused projection split must align to weight blocks.");
        }
        result.push_back({s.prefix + ".weight_scale_inv",
                          infinicore::nn::Parameter(
                              params.at("weight_scale_inv")->narrow({{static_cast<size_t>(narrow_dim), s.start / block_size_, s.size / block_size_}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
    }
    return result;
}

infinicore::Tensor FP8Block::forward(
    const ParamsMap &params, const infinicore::Tensor &input, bool has_bias, float alpha) const {
    auto weight = params.at("weight");
    auto shape = weight->shape();
    auto unpacked = infinicore::Tensor::empty(shape, infinicore::DataType::F32, weight->device());
    infinicore::op::cast_(unpacked, weight);
    auto tiles = unpacked->view({shape[0] / block_size_, block_size_, shape[1] / block_size_, block_size_});
    auto block_scales = params.at("weight_scale_inv");
    auto scales = block_scales->as_strided(tiles->shape(), {block_scales->stride(0), 0, block_scales->stride(1), 0});
    // Despite its checkpoint name, `weight_scale_inv` multiplies the FP8 values.
    infinicore::op::mul_(tiles, tiles, scales);
    auto dequantized = infinicore::Tensor::empty(shape, input->dtype(), weight->device());
    infinicore::op::cast_(dequantized, unpacked);
    auto dense_params = params;
    dense_params["weight"] = dequantized;
    return NoneQuantization::forward(dense_params, input, has_bias, alpha);
}

infinicore::Tensor FP8Block::forward_allreduce(
    const ParamsMap &params, const infinicore::Tensor &input,
    bool has_bias, infinicclComm_t communicator, float alpha) const {
    return BaseQuantization::forward_allreduce(params, input, has_bias, communicator, alpha);
}

#if INFINILM_ENABLE_FP8_MARLIN
namespace {
// InfiniCore's native Marlin operator supports FP8 as well as integer weights.
constexpr int64_t FP8_E4M3FN_ID = 2814749767172868LL;

class FP8Marlin final : public FP8Block {
public:
    FP8Marlin(const nlohmann::json &config, size_t n) : FP8Block(config), n_(n) {}
    infinicore::Tensor forward(const ParamsMap &params, const infinicore::Tensor &input,
                               bool bias, float alpha) const override {
        if (alpha != 1.0f) {
            throw std::runtime_error("FP8 Marlin currently requires linear `alpha=1`.");
        }
        auto contiguous = input->is_contiguous() ? input : input->contiguous();
        auto shape = input->shape();
        const size_t k = shape.back(), m = input->numel() / k;
        auto output = infinicore::Tensor::empty({m, n_}, input->dtype(), input->device());
        auto weight = params.at("qweight"), scales = params.at("scales"), empty = params.at("empty");
        infinicore::op::awq_marlin_gemm_(output, contiguous->view({m, k}), weight,
                                         empty, scales, empty, empty, empty, empty, empty,
                                         FP8_E4M3FN_ID, true, false, true, false);
        if (bias) {
            infinicore::op::add_(output, output, params.at("bias")->as_strided({m, n_}, {0, 1}));
        }
        shape.back() = n_;
        return output->view(shape);
    }
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &, const infinicore::Device &, int) const override { return nullptr; }
    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &,
        const std::vector<SplitInfo> &, int, int, int, int) const override { return {}; }

private:
    size_t n_;
};
} // namespace
#endif

std::shared_ptr<BaseQuantization> FP8Block::process_weights_after_loading(
    ParamsMap &params, const infinicore::Device &device, int) const {
    const auto backend = get_or<std::string>("fp8_backend", "compatibility");
    if (backend == "compatibility") {
        return nullptr;
    }
    if (backend != "marlin" || device.getType() != infinicore::Device::Type::NVIDIA) {
        throw std::runtime_error("FP8 backend must be `compatibility` or NVIDIA `marlin`.");
    }
#if INFINILM_ENABLE_FP8_MARLIN
    if (activation_dtype_ != infinicore::DataType::BF16 && activation_dtype_ != infinicore::DataType::F16) {
        throw std::runtime_error("FP8 Marlin requires BF16 or FP16 activations.");
    }
    auto weight = params.at("weight");
    const size_t n = weight->size(0), k = weight->size(1);
    // Pack bytes without requantization. Reuse the existing GPU Marlin repacker.
    auto packed = infinicore::Tensor::from_blob(weight->data(), {n, k / 4}, infinicore::DataType::I32, device)
                      ->permute({1, 0})
                      ->contiguous();
    auto empty = marlin::make_empty_i32(device);
    auto repacked = marlin::gptq_marlin_repack(packed, empty, k, n, 8);
    // Reuse original storage, also held by fused-projection checkpoint aliases.
    // Keeping `weight` owns this memory; `qweight` is its packed runtime view.
    auto qweight = infinicore::Tensor::from_blob(weight->data(), repacked->shape(), infinicore::DataType::I32, device);
    qweight->copy_from(repacked);

    auto cpu_scales = params.at("weight_scale_inv")->to(infinicore::Device::cpu())->contiguous();
    infinicore::context::syncStream();
    const auto *src = reinterpret_cast<const float *>(cpu_scales->data());
    std::vector<float> expanded(k / block_size_ * n);
    const float exponent_bias = std::ldexp(1.0f, activation_dtype_ == infinicore::DataType::BF16 ? 120 : 8);
    for (size_t group = 0; group < k / block_size_; ++group) {
        for (size_t row = 0; row < n; ++row) {
            expanded[group * n + row] = src[(row / block_size_) * (k / block_size_) + group] * exponent_bias;
        }
    }
    auto scale_f32 = infinicore::Tensor::from_blob(expanded.data(), {k / block_size_, n}, infinicore::DataType::F32, infinicore::Device::cpu())->to(device);
    auto scales = infinicore::Tensor::empty(scale_f32->shape(), activation_dtype_, device);
    infinicore::op::cast_(scales, scale_f32);
    params["qweight"] = qweight;
    params["scales"] = marlin::permute_scales(scales, k, n, block_size_);
    params["empty"] = empty;
    infinicore::context::syncStream();
    return std::make_shared<FP8Marlin>(get_config(), n);
#else
    throw std::runtime_error("FP8 Marlin requires an InfiniCore build with Marlin support.");
#endif
}

} // namespace infinilm::quantization
