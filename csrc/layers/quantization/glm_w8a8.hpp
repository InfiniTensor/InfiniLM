#pragma once
#include "base_quantization.hpp"

namespace infinilm::quantization {

// Dense GLM W8A8 checkpoint/runtime layout. We retain [out, in] so CUINFER
// scaled_mm can consume checkpoint shards directly with trans_weight=true.
class GlmW8A8 final : public BaseQuantization {
public:
    explicit GlmW8A8(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {}

    QuantScheme get_quant_scheme() const override { return QuantScheme::GLM_W8A8; }
    int get_fused_split_dim() const override { return 0; }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads, const infinicore::DataType &dtype,
        bool bias) const override;

    infinicore::Tensor forward(
        const ParamsMap &params, const infinicore::Tensor &input,
        bool has_bias, float alpha = 1.0f) const override;

    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits, int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;
};
} // namespace infinilm::quantization
