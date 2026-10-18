#pragma once
#include "../../weights/weights.hpp"

#include "base_quantization.hpp"
namespace infinilm::quantization {

class CompressedTensors : public BaseQuantization {
public:
    explicit CompressedTensors(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {
        if (quant_config.value("format", std::string{}) != "int-quantized" || !quant_config.contains("config_groups") || !quant_config.at("config_groups").is_object() || quant_config.at("config_groups").empty()) {
            throw std::invalid_argument("W8A8 requires int-quantized config_groups");
        }
        for (const auto &group : quant_config.at("config_groups")) {
            const auto check = [](const nlohmann::json &q, const char *strategy, bool dynamic) {
                return q.is_object() && q.value("num_bits", 0) == 8 && q.value("type", std::string{}) == "int" && q.value("strategy", std::string{}) == strategy && q.value("symmetric", false) && q.value("dynamic", false) == dynamic;
            };
            if (!group.contains("weights") || !group.contains("input_activations") || !check(group.at("weights"), "channel", false) || !check(group.at("input_activations"), "token", true) || (group.contains("output_activations") && !group.at("output_activations").is_null())) {
                throw std::invalid_argument("W8A8 requires symmetric channel weights and dynamic token activations");
            }
        }
    }

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
