#include "gptq.hpp"
#include "gptq_marlin.hpp"
#include "gptq_qy.hpp"
#include "marlin_support.hpp"
#include "marlin_utils.hpp"

namespace infinilm::quantization {

std::vector<ParamDescriptor> GPTQ::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinicore::DataType &dtype,
    bool bias) const {

    // GPTQ weight layout is transposed: qweight [in_features/8, out_features]
    // ColumnParallel (split_dim=0, split output) → GPTQ tp_dim=1
    // RowParallel    (split_dim=1, split input)  → GPTQ tp_dim=0
    int gptq_tp_dim = (split_dim >= 0) ? (1 - split_dim) : -1;
    int group_size = get_group_size();

    std::vector<ParamDescriptor> descs;
    descs.push_back({"qweight", {in_features / 8, out_features}, infinicore::DataType::I32, gptq_tp_dim, tp_rank, tp_size});
    descs.push_back({"qzeros", {in_features / group_size, out_features / 8}, infinicore::DataType::I32, gptq_tp_dim, tp_rank, tp_size});
    descs.push_back({"scales", {in_features / group_size, out_features}, dtype, gptq_tp_dim, tp_rank, tp_size});
    const bool row_parallel = split_dim == 1;
    descs.push_back({"g_idx", {in_features}, infinicore::DataType::I32, row_parallel ? 0 : -1, row_parallel ? tp_rank : 0, row_parallel ? tp_size : 1});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, -1, 0, 1});
    }
    return descs;
}

infinicore::Tensor GPTQ::forward(
    const ParamsMap & /*params*/,
    const infinicore::Tensor & /*input*/,
    bool /*has_bias*/,
    float /*alpha*/) const {
    throw std::runtime_error("GPTQ_W4A16 must be converted to GPTQ_QY before forward pass. "
                             "Call process_weights_after_loading() first.");
}

std::shared_ptr<BaseQuantization> GPTQ::process_weights_after_loading(
    ParamsMap &params,
    const infinicore::Device &device,
    int split_dim) const {

    if (device.getType() == infinicore::Device::Type::QY) {
        return GPTQ_QY::convert_from_gptq(params, device, get_config());
    }

    if (device.getType() == infinicore::Device::Type::NVIDIA) {
#if INFINILM_ENABLE_MARLIN
        const int bits = get_or<int>("bits", get_or<int>("w_bit", 4));
        const bool is_sym = get_or<bool>("sym", true);
        if (bits == 4 && is_sym) {
            auto qweight = params.at("qweight");
            const size_t input_size_per_partition = qweight->size(0) * get_packing_num();
            const size_t output_size_per_partition = qweight->size(1);
            const int group_size = get_group_size();
            if (marlin::supports_shape(input_size_per_partition, output_size_per_partition, group_size)) {
                const bool desc_act = get_or<bool>("desc_act", false);
                const bool row_parallel = split_dim == 1;
                const bool is_k_full = (!desc_act) || (desc_act && !row_parallel);

                if (desc_act) {
                    infinicore::Tensor perm;
                    params["g_idx"] = marlin::sort_g_idx(params.at("g_idx"), perm);
                    params["perm"] = perm;
                } else {
                    params["g_idx"] = marlin::make_empty_i32(device);
                    params["perm"] = marlin::make_empty_i32(device);
                }

                params["qweight"] = marlin::gptq_marlin_repack(
                    qweight,
                    params.at("perm"),
                    input_size_per_partition,
                    output_size_per_partition,
                    bits);
                params["scales"] = marlin::permute_scales(
                    params.at("scales"),
                    input_size_per_partition,
                    output_size_per_partition,
                    group_size);
                params["qzeros"] = marlin::make_empty_i32(device);
                params["global_scales"] = marlin::make_empty_i32(device);

                return std::make_shared<GPTQMarlin>(
                    get_config(),
                    input_size_per_partition,
                    output_size_per_partition,
                    is_k_full);
            }
        }
#else
        (void)split_dim;
#endif
    }
    return std::const_pointer_cast<BaseQuantization>(shared_from_this());
}

std::vector<SplitParam> GPTQ::split_params(
    const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
    const std::vector<SplitInfo> &splits,
    int /*narrow_dim*/,
    int tp_rank, int tp_size, int tp_num_heads) const {

    // GPTQ parameters have output dimension on dim1.
    int fused_dim = get_fused_split_dim();
    std::vector<SplitParam> result;
    auto qw_it = params.find("qweight");
    auto qz_it = params.find("qzeros");
    auto sc_it = params.find("scales");
    auto gidx_it = params.find("g_idx");
    auto bias_it = params.find("bias");

    for (const auto &s : splits) {
        result.push_back({s.prefix + ".qweight",
                          infinicore::nn::Parameter(
                              qw_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".qzeros",
                          infinicore::nn::Parameter(
                              qz_it->second->narrow({{static_cast<size_t>(fused_dim), s.start / 8, s.size / 8}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".scales",
                          infinicore::nn::Parameter(
                              sc_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".g_idx",
                          infinicore::nn::Parameter(
                              gidx_it->second->narrow({{0, 0, gidx_it->second->size(0)}}),
                              0, 0, 1, 0)});
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
