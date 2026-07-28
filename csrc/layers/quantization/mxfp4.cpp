#include "mxfp4.hpp"

#include "none_quantization.hpp"

#include <infinicore/ops/mxfp4_dequantize.hpp>

#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> MXFP4::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {
    if (in_features % 32 != 0) {
        throw std::runtime_error("MXFP4: in_features must be divisible by 32");
    }
    output_dtype_ = dtype;

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features / 2}, infinicore::DataType::U8, split_dim, tp_rank, tp_size});
    descs.push_back({"weight_scale", {out_features, in_features / 32}, infinicore::DataType::U8, split_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim >= 0 ? 0 : -1, split_dim >= 0 ? tp_rank : 0, split_dim >= 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor MXFP4::forward(
    const ParamsMap &,
    const infinicore::Tensor &,
    bool,
    float) const {
    throw std::runtime_error(
        "MXFP4: weights must be processed before the first forward pass");
}

std::vector<SplitParam> MXFP4::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {
    std::vector<SplitParam> result;
    const auto &weight = params.at("weight");
    const auto &weight_scale = params.at("weight_scale");
    const auto bias_it = params.find("bias");

    for (const auto &split : splits) {
        result.push_back({split.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight->narrow({{static_cast<size_t>(narrow_dim), split.start, split.size}}),
                              narrow_dim, tp_rank, tp_size, split.num_shards)});
        result.push_back({split.prefix + ".weight_scale",
                          infinicore::nn::Parameter(
                              weight_scale->narrow({{static_cast<size_t>(narrow_dim), split.start, split.size}}),
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

std::shared_ptr<BaseQuantization> MXFP4::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &,
    int) const {
    const auto weight_it = params.find("weight");
    const auto scale_it = params.find("weight_scale");
    if (weight_it == params.end() || scale_it == params.end()) {
        throw std::runtime_error(
            "MXFP4: post-load processing requires weight and weight_scale");
    }
    auto dequantized = infinicore::op::mxfp4_dequantize(
        weight_it->second, scale_it->second, output_dtype_);
    params["weight"] = std::move(dequantized);
    params.erase("weight_scale");
    return std::make_shared<NoneQuantization>();
}

} // namespace infinilm::quantization
