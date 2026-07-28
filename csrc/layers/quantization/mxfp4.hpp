#pragma once

#include "base_quantization.hpp"

namespace infinilm::quantization {

class MXFP4 : public BaseQuantization {
public:
    explicit MXFP4(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {}

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::MXFP4_W4A16;
    }

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

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

private:
    mutable infinicore::DataType output_dtype_{infinicore::DataType::BF16};
};

} // namespace infinilm::quantization
