#include "awq.hpp"
#include "awq_marlin.hpp"
#include "infinicore/ops/linear_w4a16_awq.hpp"
#include "marlin_support.hpp"
#include "marlin_utils.hpp"
#include <optional>

namespace infinilm::quantization {

std::vector<ParamDescriptor> AWQ::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    // AWQ weight layout is transposed relative to the standard [out, in]:
    //   qweight: [in_features, out_features / packing_num]
    // So the TP split dimension for AWQ is the "other" dimension:
    //   ColumnParallel (split_dim=0, split output) → AWQ tp_dim=1
    //   RowParallel    (split_dim=1, split input)  → AWQ tp_dim=0
    int awq_tp_dim = (split_dim >= 0) ? (1 - split_dim) : -1;
    int group_size = get_group_size();
    int packing_num = get_packing_num();

    std::vector<ParamDescriptor> descs;
    descs.push_back({"qweight", {in_features, out_features / packing_num}, infinicore::DataType::I32, awq_tp_dim, tp_rank, tp_size});
    descs.push_back({"scales", {in_features / group_size, out_features}, dtype, awq_tp_dim, tp_rank, tp_size});
    descs.push_back({"qzeros", {in_features / group_size, out_features / packing_num}, infinicore::DataType::I32, awq_tp_dim, tp_rank, tp_size});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor AWQ::forward(
    const ParamsMap &params,
    const infinicore::Tensor &input,
    bool has_bias,
    float /*alpha*/) const {

    auto input_contiguous = input->is_contiguous() ? input : input->contiguous();
    auto qweight = params.at("qweight");
    auto scales = params.at("scales");
    auto qzeros = params.at("qzeros");

    std::optional<infinicore::Tensor> bias_opt;
    if (has_bias) {
        bias_opt = params.at("bias");
    }

    return infinicore::op::linear_w4a16_awq(input_contiguous->contiguous(), qweight, scales, qzeros, bias_opt);
}

std::shared_ptr<BaseQuantization> AWQ::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int /*split_dim*/) const {
    if (device.getType() != infinicore::Device::Type::NVIDIA) {
        return nullptr;
    }

#if INFINILM_ENABLE_MARLIN
    const int bits = get_or<int>("bits", get_or<int>("w_bit", 4));
    if (bits != 4) {
        return nullptr;
    }

    auto qweight = params.at("qweight");
    const size_t input_size_per_partition = qweight->size(0);
    const size_t output_size_per_partition = qweight->size(1) * get_packing_num();
    const int group_size = get_group_size();
    if (!marlin::supports_shape(input_size_per_partition, output_size_per_partition, group_size)) {
        return nullptr;
    }

    params["qweight"] = marlin::awq_marlin_repack(
        qweight,
        input_size_per_partition,
        output_size_per_partition,
        bits);
    params["scales"] = marlin::permute_scales(
        params.at("scales"),
        input_size_per_partition,
        output_size_per_partition,
        group_size);
    params["qzeros"] = marlin::awq_to_marlin_zero_points(
        params.at("qzeros"),
        input_size_per_partition / static_cast<size_t>(group_size == -1 ? input_size_per_partition : group_size),
        output_size_per_partition,
        bits);
    params["g_idx"] = marlin::make_empty_i32(device);
    params["perm"] = marlin::make_empty_i32(device);
    params["a_scales"] = marlin::make_empty_i32(device);
    params["global_scales"] = marlin::make_empty_i32(device);

    return std::make_shared<AWQMarlin>(get_config(), input_size_per_partition, output_size_per_partition);
#else
    (void)params;
    return nullptr;
#endif
}

std::vector<SplitParam> AWQ::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int /*narrow_dim*/,
    int tp_rank, int tp_size, int tp_num_heads) const {

    // AWQ parameters have output dimension on dim1, so fused split is on dim1.
    int fused_dim = get_fused_split_dim();
    int packing_num = get_packing_num();
    std::vector<SplitParam> result;
    auto qw_it = params.find("qweight");
    auto sc_it = params.find("scales");
    auto qz_it = params.find("qzeros");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        // qweight: narrow along fused_dim, divide size by packing_num
        result.push_back({s.prefix + ".qweight",
                          infinicore::nn::Parameter(
                              qw_it->second->narrow({{static_cast<size_t>(fused_dim), s.start / packing_num, s.size / packing_num}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        // scales: narrow along fused_dim
        result.push_back({s.prefix + ".scales",
                          infinicore::nn::Parameter(
                              sc_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        // qzeros: narrow along fused_dim, divide size by packing_num
        result.push_back({s.prefix + ".qzeros",
                          infinicore::nn::Parameter(
                              qz_it->second->narrow({{static_cast<size_t>(fused_dim), s.start / packing_num, s.size / packing_num}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
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
