#pragma once
#include "../../weights/weights.hpp"
#include "base_quantization.hpp"
namespace infinilm::quantization {

class AWQ : public BaseQuantization {
public:
    explicit AWQ(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::AWQ_W4A16;
    };

    int get_packing_num() const {
        const int bits = get_or<int>("bits", get_or<int>("w_bit", 4));
        if (bits != 4) {
            throw std::invalid_argument("W4 packing requires exactly four bits");
        }
        return 8;
    }

    int get_group_size() const {
        return get_or<int>("group_size", get_or<int>("q_group_size", 128));
    }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinilm::runtime::DataType &dtype,
        bool bias) const override;

    int get_fused_split_dim() const override { return 1; }

    size_t get_logical_dim_size(size_t raw_size) const override {
        return raw_size * get_packing_num();
    }

    infinilm::runtime::Tensor forward(
        infinilm::runtime::Runtime &runtime,
        const ParamsMap &params,
        const infinilm::runtime::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinilm::weights::Weight> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    std::shared_ptr<const W4Execution> prepare_w4_after_loading(
        infinilm::runtime::Runtime &, const ParamsMap &, int split_dim,
        bool retained_global_scale_table,
        std::optional<int64_t> expected_non_desc_group_offset = std::nullopt) const override;

    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        infinilm::runtime::Runtime &runtime, ParamsMap &params, bool pre_transpose,
        int split_dim = -1) const override;
};

} // namespace infinilm::quantization
