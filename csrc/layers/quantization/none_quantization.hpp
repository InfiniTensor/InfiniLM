#pragma once

#include "base_quantization.hpp"
namespace infinilm::quantization {

class NoneQuantization : public BaseQuantization {
public:
    explicit NoneQuantization(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config){};

    NoneQuantization();

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::NONE;
    };

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
        int tp_rank, int tp_size, int tp_num_heads) const override;

    // Ascend: pre-pack weight to [IC, OC] after loading to skip runtime permute.
    // Returns shared_from_this() only on Ascend; nullptr otherwise (no-op).
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &params,
        const infinicore::Device &device,
        int split_dim = -1) const override;

private:
    mutable bool weight_prepacked_ = false; // true when weight was pre-packed for Ascend
};

} // namespace infinilm::quantization
