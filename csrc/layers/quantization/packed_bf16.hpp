#pragma once

#include "base_quantization.hpp"

namespace infinilm::quantization {

// Lossless BF16 layout: weights are pre-transposed offline from [N, K] to
// [K, N]. This is not quantization; activation and weight dtypes are unchanged.
class PackedBF16 : public BaseQuantization {
public:
    explicit PackedBF16(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {}

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::PACKED_BF16;
    }

    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params, const infinicore::Device &device,
        int split_dim = -1) const override;

    int get_fused_split_dim() const override;
    bool use_full_layout() const;

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features,
        size_t out_features,
        int split_dim,
        int tp_rank,
        int tp_size,
        int tp_num_heads,
        const infinicore::DataType &dtype,
        bool bias) const override;

    infinicore::Tensor forward(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        float alpha = 1.0f) const override;

    infinicore::Tensor forward_allreduce(
        const ParamsMap &params,
        const infinicore::Tensor &input,
        bool has_bias,
        infinicclComm_t communicator,
        float alpha = 1.0f) const override;

    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits,
        int narrow_dim,
        int tp_rank,
        int tp_size,
        int tp_num_heads) const override;
};

} // namespace infinilm::quantization
