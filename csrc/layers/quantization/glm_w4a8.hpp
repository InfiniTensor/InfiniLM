#pragma once
#include "base_quantization.hpp"

namespace infinilm::quantization {

class GlmW4A8Runtime;

// Loader-side GLM W4A8 quantization. Checkpoint weight layout is [out, in/2]
// with two 4-bit signed weights packed in one int8 byte along K.
class GlmW4A8 : public BaseQuantization {
public:
    explicit GlmW4A8(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {}

    QuantScheme get_quant_scheme() const override { return QuantScheme::GLM_W4A8; }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    // Initial checkpoint layout has output dimension on dim0.
    int get_fused_split_dim() const override { return 0; }

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

    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params,
        const infinicore::Device &device,
        int split_dim = -1) const override;
};

// Runtime GLM W4A8 layout for CUINFER scaled_mm_w4a8: weight [in, out/2].
class GlmW4A8Runtime : public BaseQuantization {
public:
    explicit GlmW4A8Runtime(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {}

    QuantScheme get_quant_scheme() const override { return QuantScheme::GLM_W4A8_RUNTIME; }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    // Runtime packed layout has logical output dimension on dim1, scaled by 2.
    int get_fused_split_dim() const override { return 1; }
    size_t get_logical_dim_size(size_t raw_size) const override { return raw_size * 2; }

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
};

} // namespace infinilm::quantization
