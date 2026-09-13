#include "glm_w8a8.hpp"

#include "infinicore/ops/dynamic_scaled_int8_quant.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include "infinicore/ops/scaled_mm_w8a8.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> GlmW8A8::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/, const infinicore::DataType &dtype,
    bool bias) const {
    std::vector<ParamDescriptor> result;
    if (runtime_layout_) {
        const int runtime_split_dim = split_dim >= 0 ? 1 - split_dim : -1;
        result.push_back({"weight", {in_features, out_features}, infinicore::DataType::I8, runtime_split_dim, tp_rank, tp_size});
    } else {
        result.push_back({"weight", {out_features, in_features}, infinicore::DataType::I8, split_dim, tp_rank, tp_size});
    }
    const int scale_split_dim = split_dim == 0 ? 0 : -1;
    result.push_back({"weight_scale", {out_features, 1}, infinicore::DataType::F32, scale_split_dim, scale_split_dim == 0 ? tp_rank : 0, scale_split_dim == 0 ? tp_size : 1});
    if (bias) {
        result.push_back({"bias", {out_features}, dtype, split_dim == 0 ? 0 : -1, split_dim == 0 ? tp_rank : 0, split_dim == 0 ? tp_size : 1});
    }
    return result;
}

infinicore::Tensor GlmW8A8::forward(
    const ParamsMap &params, const infinicore::Tensor &input,
    bool has_bias, float alpha) const {
    auto weight = params.at("weight");
    if (!runtime_layout_) {
        throw std::runtime_error("GlmW8A8 weights must be converted to runtime layout after loading");
    }
    auto weight_scale = params.at("weight_scale");
    if (weight->ndim() != 2 || weight->dtype() != infinicore::DataType::I8) {
        throw std::runtime_error("GlmW8A8 expects int8 runtime weight [in,out]");
    }
    const size_t in_features = weight->size(0), out_features = weight->size(1);
    if (weight_scale->ndim() != 2 || weight_scale->size(0) != out_features || weight_scale->size(1) != 1 || weight_scale->dtype() != infinicore::DataType::F32) {
        throw std::runtime_error("GlmW8A8 expects float32 weight_scale [out,1]");
    }

    auto x = input->is_contiguous() ? input : input->contiguous();
    if (x->size(x->ndim() - 1) != in_features) {
        throw std::runtime_error("GlmW8A8 input hidden size mismatch");
    }
    auto shape = x->shape();
    size_t m = 1;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
        m *= shape[i];
    }
    auto x2d = x->ndim() == 2 ? x : x->view({m, in_features});
    auto x_i8 = infinicore::Tensor::empty({m, in_features}, infinicore::DataType::I8, x->device());
    auto x_scale = infinicore::Tensor::empty({m, 1}, infinicore::DataType::F32, x->device());
    infinicore::op::dynamic_scaled_int8_quant_(x_i8, x2d, x_scale);

    auto effective_scale = weight_scale;
    if (std::fabs(alpha - 1.0f) > 1e-7f) {
        effective_scale = infinicore::op::mul_scalar(weight_scale, static_cast<double>(alpha));
    }
    std::optional<infinicore::Tensor> bias;
    if (has_bias) {
        bias = params.at("bias");
    }
    auto out = infinicore::Tensor::empty({m, out_features}, x->dtype(), x->device());
    infinicore::op::scaled_mm_w8a8_(out, x_i8, weight, x_scale, effective_scale, bias, false);
    if (shape.size() == 2) {
        return out;
    }
    shape.back() = out_features;
    return out->view(shape);
}

std::vector<SplitParam> GlmW8A8::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits, int /*narrow_dim*/,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {
    std::vector<SplitParam> result;
    const auto &weight = params.at("weight");
    const auto &scale = params.at("weight_scale");
    auto bias = params.find("bias");
    for (const auto &s : splits) {
        const int weight_dim = runtime_layout_ ? 1 : 0;
        result.push_back({s.prefix + ".weight", infinicore::nn::Parameter(
                                                    weight->narrow({{static_cast<size_t>(weight_dim), s.start, s.size}}), weight_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".weight_scale", infinicore::nn::Parameter(
                                                          scale->narrow({{0, s.start, s.size}}), 0, tp_rank, tp_size, s.num_shards)});
        if (bias != params.end()) {
            result.push_back({s.prefix + ".bias", infinicore::nn::Parameter(
                                                      bias->second->narrow({{0, s.start, s.size}}), 0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> GlmW8A8::process_weights_after_loading(
    ParamsMap &params, const infinicore::Device &device,
    int /*split_dim*/) const {
    (void)device;
    if (runtime_layout_) {
        return nullptr;
    }
    auto weight = params.at("weight");
    params["weight"] = weight->permute({1, 0})->contiguous();
    return std::make_shared<GlmW8A8>(get_config(), true);
}
} // namespace infinilm::quantization
