#include "gptq_qy.hpp"

#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> GPTQ_QY::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {
    const int tp_dim = split_dim >= 0 ? 1 - split_dim : -1;
    const int group_size = get_group_size();
    if (weight_bits() != 4 || group_size <= 0) {
        throw std::invalid_argument("GPTQ QY parameter layout requires 4-bit weights and a positive group size.");
    }
    std::vector<ParamDescriptor> descs{
        {"qweight", {in_features / 2, out_features}, infinicore::DataType::kUInt8, tp_dim, tp_rank, tp_size},
        {"qzeros", {in_features / group_size, out_features}, dtype, tp_dim, tp_rank, tp_size},
        {"scales", {in_features / group_size, out_features}, dtype, tp_dim, tp_rank, tp_size},
        {"g_idx", {in_features}, infinicore::DataType::kInt32, -1, 0, 1}};
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor GPTQ_QY::forward(
    const ParamsMap &,
    const infinicore::Tensor &,
    bool,
    float) const {
    throw std::runtime_error(
        "GPTQ QY quantization is unsupported because InfiniLM has no InfiniOps-backed QY GEMM path.");
}

std::vector<SplitParam> GPTQ_QY::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int /*narrow_dim*/,
    int tp_rank, int tp_size, int /*tp_num_heads*/) const {
    const size_t fused_dim = get_fused_split_dim();
    std::vector<SplitParam> result;
    for (const auto &s : splits) {
        for (const auto *name : {"qweight", "qzeros", "scales"}) {
            result.push_back({s.prefix + "." + name,
                              infinicore::nn::Parameter(
                                  params.at(name)->narrow({{fused_dim, s.start, s.size}}),
                                  fused_dim, tp_rank, tp_size, s.num_shards)});
        }
        const auto &g_idx = params.at("g_idx");
        result.push_back({s.prefix + ".g_idx",
                          infinicore::nn::Parameter(
                              g_idx->narrow({{0, 0, g_idx->size(0)}}), 0, 0, 1, 0)});
        if (const auto bias = params.find("bias"); bias != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinicore::nn::Parameter(
                                  bias->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

std::shared_ptr<BaseQuantization> GPTQ_QY::convert_from_gptq(
    ParamsMap &,
    const infinicore::Device &,
    const nlohmann::json &) {
    throw std::runtime_error(
        "GPTQ QY conversion is unsupported because InfiniLM has no InfiniOps-backed QY GEMM path.");
}

} // namespace infinilm::quantization
