#include "gptq.hpp"
#include "../../weights/weights.hpp"
#include "w4_execution.hpp"
#include <stdexcept>

namespace infinilm::quantization {

std::vector<ParamDescriptor> GPTQ::get_param_layout(
    size_t in_features, size_t out_features,
    int split_dim, int tp_rank, int tp_size,
    int /*tp_num_heads*/,
    const infinilm::runtime::DataType &dtype,
    bool bias) const {

    // GPTQ weight layout is transposed: qweight [in_features/8, out_features]
    // ColumnParallel (split_dim=0, split output) → GPTQ tp_dim=1
    // RowParallel    (split_dim=1, split input)  → GPTQ tp_dim=0
    int gptq_tp_dim = (split_dim >= 0) ? (1 - split_dim) : -1;
    int group_size = get_group_size();
    if (get_or<int>("bits", get_or<int>("w_bit", 4)) != 4 || get_packing_num() != 8 || !(group_size == -1 || group_size == 32 || group_size == 64 || group_size == 128) || !in_features || !out_features || in_features % 8 || out_features % 8 || split_dim < -1 || split_dim > 1 || tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size || (group_size > 0 && in_features % group_size)) {
        throw std::invalid_argument("W4 raw checkpoint layout/TP/group/packing invalid");
    }
    const size_t groups = group_size == -1 ? 1 : in_features / size_t(group_size);
    const bool retain_groups = group_size == -1 || get_or<bool>("desc_act", false);
    const int table_tp_dim = split_dim == 1 && retain_groups ? -1 : gptq_tp_dim;
    const int table_rank = table_tp_dim < 0 ? 0 : tp_rank;
    const int table_size = table_tp_dim < 0 ? 1 : tp_size;

    std::vector<ParamDescriptor> descs;
    descs.push_back({"qweight", {in_features / 8, out_features}, infinilm::runtime::DataType::kInt32, gptq_tp_dim, tp_rank, tp_size});
    descs.push_back({"qzeros", {groups, out_features / 8}, infinilm::runtime::DataType::kInt32, table_tp_dim, table_rank, table_size});
    descs.push_back({"scales", {groups, out_features}, dtype, table_tp_dim, table_rank, table_size});
    const bool row_parallel = split_dim == 1;
    descs.push_back({"g_idx", {in_features}, infinilm::runtime::DataType::kInt32, row_parallel ? 0 : -1, row_parallel ? tp_rank : 0, row_parallel ? tp_size : 1});
    if (bias) {
        descs.push_back({"bias", {out_features}, dtype, split_dim == 0 ? 0 : -1, split_dim == 0 ? tp_rank : 0, split_dim == 0 ? tp_size : 1});
    }
    return descs;
}

std::shared_ptr<const W4Execution> GPTQ::prepare_w4_after_loading(
    infinilm::runtime::Runtime &runtime, const ParamsMap &raw, int split_dim,
    bool retained_global_scale_table,
    std::optional<int64_t> expected_non_desc_group_offset) const {
    return W4Execution::prepare(runtime, raw, get_config(), split_dim, retained_global_scale_table, expected_non_desc_group_offset);
}

infinilm::runtime::Tensor GPTQ::forward(
    infinilm::runtime::Runtime &,
    const ParamsMap & /*params*/,
    const infinilm::runtime::Tensor & /*input*/,
    bool /*has_bias*/,
    float /*alpha*/) const {
    throw std::runtime_error(
        "Raw GPTQ forward requires an explicitly prepared and published BaseLinear W4 binding.");
}

std::shared_ptr<BaseQuantization> GPTQ::process_weights_after_loading(
    infinilm::runtime::Runtime &, ParamsMap &, bool, int) const {
    throw std::runtime_error(
        "GPTQ device-only finalize is unsupported; use explicit Runtime W4 stage/publish.");
}

std::vector<SplitParam> GPTQ::split_params(
    const std::unordered_map<std::string, infinilm::weights::Weight> &params,
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

    if (params.find("qweight") == params.end() || params.find("scales") == params.end() || params.find("qzeros") == params.end() || params.find("g_idx") == params.end()) {
        throw std::invalid_argument("W4 fused raw fields missing");
    }
    if (get_or<int>("bits", get_or<int>("w_bit", 4)) != 4 || !params.at("qweight") || !params.at("scales") || !params.at("qzeros") || !params.at("g_idx") || params.at("qweight")->ndim() != 2 || params.at("scales")->ndim() != 2 || params.at("qzeros")->ndim() != 2) {
        throw std::invalid_argument("W4 fused raw tensor layout invalid");
    }
    const size_t logical_n = params.at("scales")->size(1);
    for (const auto &s : splits) {
        if (s.start % 8 || s.size % 8 || !s.size || s.start > logical_n || s.size > logical_n - s.start) {
            throw std::invalid_argument("W4 fused raw output slice must align packed eight-nibble boundary");
        }
    }
    for (const auto &s : splits) {
        result.push_back({s.prefix + ".qweight",
                          infinilm::weights::Weight(
                              qw_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".qzeros",
                          infinilm::weights::Weight(
                              qz_it->second->narrow({{static_cast<size_t>(fused_dim), s.start / 8, s.size / 8}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".scales",
                          infinilm::weights::Weight(
                              sc_it->second->narrow({{static_cast<size_t>(fused_dim), s.start, s.size}}),
                              fused_dim, tp_rank, tp_size, s.num_shards)});
        result.push_back({s.prefix + ".g_idx",
                          infinilm::weights::Weight(
                              gidx_it->second->narrow({{0, 0, gidx_it->second->size(0)}}),
                              0, 0, 1, 0)});
        if (bias_it != params.end()) {
            result.push_back({s.prefix + ".bias",
                              infinilm::weights::Weight(
                                  bias_it->second->narrow({{0, s.start, s.size}}),
                                  0, tp_rank, tp_size, s.num_shards)});
        }
    }
    return result;
}

} // namespace infinilm::quantization
