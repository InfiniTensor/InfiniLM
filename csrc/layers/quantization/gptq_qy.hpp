#pragma once

#include "base_quantization.hpp"

namespace infinilm::quantization {

class GPTQ_QY : public BaseQuantization {
public:
    explicit GPTQ_QY(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {}

    QuantScheme get_quant_scheme() const override { return QuantScheme::GPTQ_W4A16_QY; }

    int get_packing_num() const { return 32 / weight_bits(); }
    int get_group_size() const { return get_or<int>("group_size", 128); }
    int weight_bits() const { return get_or<int>("bits", 4); }
    bool desc_act() const { return get_or<bool>("desc_act", false); }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    int get_fused_split_dim() const override { return 1; }

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;

    static std::shared_ptr<BaseQuantization> convert_from_gptq(
        ParamsMap &params,
        const infinicore::Device &device,
        const nlohmann::json &quant_config);
};

} // namespace infinilm::quantization
