#pragma once
#include "../../weights/weights.hpp"

#include "base_quantization.hpp"
namespace infinilm::quantization {

class CompressedTensors : public BaseQuantization {
public:
    explicit CompressedTensors(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {};

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::COMPRESSED_TENSOR_W8A8I8;
    };

    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features,
        int split_dim, int tp_rank, int tp_size,
        int tp_num_heads,
        const infinilm::runtime::DataType &dtype,
        bool bias) const override;

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
};

} // namespace infinilm::quantization
