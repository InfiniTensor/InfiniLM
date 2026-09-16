#include "w4a8.hpp"

#include <infinicore/ops/linear_w4a8.hpp>

#include <optional>
#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> W4A8::get_param_layout(
    size_t in_features,
    size_t out_features,
    int split_dim,
    int tp_rank,
    int tp_size,
    int,
    const infinicore::DataType &dtype,
    bool bias) const {
    if (in_features % 2 != 0) {
        throw std::runtime_error("W4A8: in_features must be even");
    }
    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features / 2}, infinicore::DataType::I8, split_dim, tp_rank, tp_size});

    const int scale_split_dim = split_dim == 0 ? 0 : -1;
    descs.push_back({"weight_scale", {out_features, 1}, infinicore::DataType::F32, scale_split_dim, scale_split_dim >= 0 ? tp_rank : 0, scale_split_dim >= 0 ? tp_size : 1});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim == 0 ? 0 : -1, split_dim == 0 ? tp_rank : 0, split_dim == 0 ? tp_size : 1});
    }
    return descs;
}

infinicore::Tensor W4A8::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float alpha) const {
    std::optional<infinicore::Tensor> bias = std::nullopt;
    if (has_bias) {
        bias = params.at("bias");
    }
    return infinicore::op::linear_w4a8(
        input->contiguous(), params.at("weight"),
        params.at("weight_scale"), bias, alpha);
}

std::vector<SplitParam> W4A8::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank,
    int tp_size,
    int) const {
    if (narrow_dim != 0) {
        throw std::runtime_error(
            "W4A8: fused linear splitting is supported only on output rows");
    }
    std::vector<SplitParam> result;
    const auto &weight = params.at("weight");
    const auto &weight_scale = params.at("weight_scale");
    const auto bias_it = params.find("bias");
    for (const auto &split : splits) {
        result.push_back({split.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight->narrow({{0, split.start, split.size}}),
                              0, tp_rank, tp_size, split.num_shards)});
        result.push_back({split.prefix + ".weight_scale",
                          infinicore::nn::Parameter(
                              weight_scale->narrow({{0, split.start, split.size}}),
                              0, tp_rank, tp_size, split.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({split.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, split.start, split.size}}),
                                  0, tp_rank, tp_size, split.num_shards)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
