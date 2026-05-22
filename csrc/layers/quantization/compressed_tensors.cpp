#include "compressed_tensors.hpp"
#include "infinicore/ops/linear_w8a8i8.hpp"
#include <optional>

namespace infinilm::quantization {

std::vector<ParamDescriptor> CompressedTensors::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    std::vector<ParamDescriptor> descs;
    descs.push_back({"weight", {out_features, in_features}, infinicore::DataType::I8, split_dim, tp_rank, tp_size});
    // weight_scale is per-output-channel [out_features, 1]; always split on
    // dim0 (output dimension) for ColumnParallel, and don't split for RowParallel.
    int scale_split_dim = (split_dim == 0) ? 0 : -1;
    int scale_tp_size = (split_dim == 0) ? tp_size : 1;
    int scale_tp_rank = (split_dim == 0) ? tp_rank : 0;
    descs.push_back({"weight_scale", {out_features, 1}, infinicore::DataType::F32, scale_split_dim, scale_tp_rank, scale_tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor CompressedTensors::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float /*alpha*/) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto weight = params.at("weight");
    auto weight_scale = params.at("weight_scale");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    return infinicore::op::linear_w8a8i8(input_contiguous->contiguous(), weight, weight_scale, bias_opt);
}

std::vector<SplitParam> CompressedTensors::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int narrow_dim,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {

    std::vector<SplitParam> result;
    auto weight_it = params.find("weight");
    auto scale_it = params.find("weight_scale");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".weight",
                          infinicore::nn::Parameter(
                              weight_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".weight_scale",
                          infinicore::nn::Parameter(
                              scale_it->second->narrow({{static_cast<size_t>(narrow_dim), s.start, s.size}}),
                              narrow_dim, tp_rank, tp_size, s.num_shards)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
