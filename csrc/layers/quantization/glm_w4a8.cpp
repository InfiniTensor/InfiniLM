#include "glm_w4a8.hpp"

#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/scaled_mm_w4a8.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace infinilm::quantization {
namespace {

infinicore::Tensor repack_out_khalf_to_k_outhalf_cpu(const infinicore::Tensor &packed_weight,
                                                     const infinicore::Device &device) {
    if (packed_weight->ndim() != 2 || packed_weight->dtype() != infinicore::DataType::I8) {
        throw std::runtime_error("GlmW4A8 expects int8 weight with shape [out, in/2]");
    }
    const size_t out_features = packed_weight->size(0);
    const size_t in_half = packed_weight->size(1);
    const size_t in_features = in_half * 2;
    if ((out_features % 2) != 0) {
        throw std::runtime_error("GlmW4A8 requires even out_features for packed runtime layout");
    }

    auto src_cpu = packed_weight->to(infinicore::Device::cpu())->contiguous();
    auto dst_cpu = infinicore::Tensor::empty({in_features, out_features / 2}, infinicore::DataType::I8, infinicore::Device::cpu());
    const auto *src = reinterpret_cast<const std::uint8_t *>(src_cpu->data());
    auto *dst = reinterpret_cast<std::uint8_t *>(dst_cpu->data());

    for (size_t k = 0; k < in_features; ++k) {
        const size_t k_byte = k / 2;
        const bool k_high = (k & 1) != 0;
        for (size_t o = 0; o < out_features; ++o) {
            const std::uint8_t src_byte = src[o * in_half + k_byte];
            const std::uint8_t nibble = k_high ? ((src_byte >> 4) & 0x0f) : (src_byte & 0x0f);
            std::uint8_t &dst_byte = dst[k * (out_features / 2) + (o / 2)];
            if ((o & 1) == 0) {
                dst_byte = static_cast<std::uint8_t>((dst_byte & 0xf0) | nibble);
            } else {
                dst_byte = static_cast<std::uint8_t>((dst_byte & 0x0f) | (nibble << 4));
            }
        }
    }
    return dst_cpu->to(device);
}

infinicore::Tensor normalize_scale(const infinicore::Tensor &scale, size_t out_features, const infinicore::Device &device) {
    if (scale->dtype() != infinicore::DataType::F32) {
        throw std::runtime_error("GlmW4A8 expects float32 weight_scale");
    }
    if (scale->ndim() == 2 && scale->size(0) == out_features && scale->size(1) == 1) {
        return scale->is_contiguous() ? scale : scale->contiguous();
    }
    if (scale->ndim() == 2 && scale->size(0) == 1 && scale->size(1) == out_features) {
        auto cpu = scale->to(infinicore::Device::cpu())->contiguous();
        auto fixed_cpu = infinicore::Tensor::empty({out_features, 1}, infinicore::DataType::F32, infinicore::Device::cpu());
        auto *src = reinterpret_cast<const float *>(cpu->data());
        auto *dst = reinterpret_cast<float *>(fixed_cpu->data());
        for (size_t i = 0; i < out_features; ++i) {
            dst[i] = src[i];
        }
        return fixed_cpu->to(device);
    }
    throw std::runtime_error("GlmW4A8 expects weight_scale shape [out,1] or [1,out]");
}

infinicore::Tensor run_w4a8_forward(const ParamsMap &params,
                                    const infinicore::Tensor &input,
                                    bool has_bias,
                                    float alpha) {
    auto weight = params.at("weight");
    auto weight_scale = params.at("weight_scale");
    if (weight->ndim() != 2 || weight->dtype() != infinicore::DataType::I8) {
        throw std::runtime_error("GlmW4A8Runtime expects int8 runtime weight [in,out/2]");
    }
    const size_t in_features = weight->size(0);
    const size_t out_features = weight->size(1) * 2;
    if (weight_scale->ndim() != 2 || weight_scale->size(0) != out_features || weight_scale->size(1) != 1) {
        throw std::runtime_error("GlmW4A8Runtime expects weight_scale [out,1]");
    }

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    auto effective_weight_scale = weight_scale;
    if (std::fabs(alpha - 1.0f) > 1e-7f) {
        effective_weight_scale = infinicore::op::mul_scalar(weight_scale, static_cast<double>(alpha));
    }

    auto x = input->is_contiguous() ? input : input->contiguous();
    if (x->size(x->ndim() - 1) != in_features) {
        throw std::runtime_error("GlmW4A8Runtime input hidden size mismatch");
    }
    std::vector<size_t> original_shape = x->shape();
    size_t m = 1;
    for (size_t i = 0; i + 1 < original_shape.size(); ++i) {
        m *= original_shape[i];
    }
    auto x2d = (x->ndim() == 2) ? x : x->view({m, in_features});

    auto x_i8 = infinicore::Tensor::empty({m, in_features}, infinicore::DataType::I8, x->device());
    auto x_scale = infinicore::Tensor::empty({m, 1}, infinicore::DataType::F32, x->device());
    infinicore::op::dynamic_scaled_int8_quant_(x_i8, x2d, x_scale);

    auto out2d = infinicore::Tensor::empty({m, out_features}, x->dtype(), x->device());
    infinicore::op::scaled_mm_w4a8_(out2d, x_i8, weight, x_scale, effective_weight_scale, bias_opt, false);

    if (original_shape.size() == 2) {
        return out2d;
    }
    original_shape.back() = out_features;
    return out2d->view(original_shape);
}

std::vector<SplitParam> split_runtime_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int tp_rank, int tp_size, int tp_num_heads) {
    std::vector<SplitParam> result;
    auto w_it = params.find("weight");
    auto s_it = params.find("weight_scale");
    auto b_it = params.find("bias");
    for (const auto &s : splits) {
        if ((s.start % 2) != 0 || (s.size % 2) != 0) {
            throw std::runtime_error("GlmW4A8Runtime split requires even output offsets/sizes");
        }
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              w_it->second->narrow({{1, s.start / 2, s.size / 2}}),
                              1, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".weight_scale",
                          infinicore::nn::Parameter(
                              s_it->second->narrow({{0, s.start, s.size}}),
                              0, tp_rank, tp_size, s.num_shards)});
        if (b_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  b_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    (void)tp_num_heads;
    return result;
}

} // namespace

std::vector<ParamDescriptor> GlmW4A8::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {
    if ((in_features % 2) != 0) {
        throw std::runtime_error("GlmW4A8 requires even in_features");
    }
    std::vector<ParamDescriptor> descs;
    // Checkpoint layout. ColumnParallel splits output dim0; RowParallel splits input packed dim1.
    int weight_split_dim = split_dim;
    descs.push_back({"weight", {out_features, in_features / 2}, infinicore::DataType::I8, weight_split_dim, tp_rank, tp_size});
    int scale_split_dim = (split_dim == 0) ? 0 : -1;
    descs.push_back({"weight_scale", {out_features, 1}, infinicore::DataType::F32, scale_split_dim, scale_split_dim == 0 ? tp_rank : 0, scale_split_dim == 0 ? tp_size : 1});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor GlmW4A8::forward(const ParamsMap &, const infinicore::Tensor &, bool, float) const {
    throw std::runtime_error("GlmW4A8 must be converted to runtime layout after loading before forward");
}

std::vector<SplitParam> GlmW4A8::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int tp_num_heads) const {
    // Loader/checkpoint layout: output dimension is dim0 for both weight and scale.
    std::vector<SplitParam> result;
    auto w_it = params.find("weight");
    auto s_it = params.find("weight_scale");
    auto b_it = params.find("bias");
    (void)narrow_dim;
    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(w_it->second->narrow({{0, s.start, s.size}}), 0, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".weight_scale",
                          infinicore::nn::Parameter(s_it->second->narrow({{0, s.start, s.size}}), 0, tp_rank, tp_size, s.num_shards)});
        if (b_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(b_it->second->narrow({{0, s.start, s.size}}), 0, tp_rank, tp_size, s.num_shards)});
        }
    }
    (void)tp_num_heads;
    return result;
}

std::shared_ptr<BaseQuantization> GlmW4A8::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int /*split_dim*/) const {
    auto weight = params.at("weight");
    const size_t out_features = weight->size(0);
    params["weight"] = repack_out_khalf_to_k_outhalf_cpu(weight, device);
    params["weight_scale"] = normalize_scale(params.at("weight_scale"), out_features, device);
    return std::make_shared<GlmW4A8Runtime>(get_config());
}

std::vector<ParamDescriptor> GlmW4A8Runtime::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {
    if ((out_features % 2) != 0) {
        throw std::runtime_error("GlmW4A8Runtime requires even out_features");
    }
    std::vector<ParamDescriptor> descs;
    int weight_split_dim = split_dim >= 0 ? (1 - split_dim) : -1;
    descs.push_back({"weight", {in_features, out_features / 2}, infinicore::DataType::I8, weight_split_dim, tp_rank, tp_size});
    int scale_split_dim = (split_dim == 0) ? 0 : -1;
    descs.push_back({"weight_scale", {out_features, 1}, infinicore::DataType::F32, scale_split_dim, scale_split_dim == 0 ? tp_rank : 0, scale_split_dim == 0 ? tp_size : 1});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor GlmW4A8Runtime::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    return run_w4a8_forward(params, input, has_bias, alpha);
}

std::vector<SplitParam> GlmW4A8Runtime::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int /*narrow_dim*/,
    int tp_rank, int tp_size, int tp_num_heads) const {
    return split_runtime_params(params, splits, tp_rank, tp_size, tp_num_heads);
}

} // namespace infinilm::quantization
