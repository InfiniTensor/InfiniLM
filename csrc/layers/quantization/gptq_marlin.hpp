#pragma once

#include "base_quantization.hpp"

namespace infinilm::quantization {

class GPTQMarlin : public BaseQuantization {
public:
    GPTQMarlin(const nlohmann::json &quant_config, size_t input_size_per_partition,
               size_t output_size_per_partition, bool is_k_full)
        : BaseQuantization(quant_config),
          input_size_per_partition_(input_size_per_partition),
          output_size_per_partition_(output_size_per_partition),
          is_k_full_(is_k_full) {}

    QuantScheme get_quant_scheme() const override { return QuantScheme::GPTQ_MARLIN_W4A16; }

    std::vector<ParamDescriptor> get_param_layout(
        size_t, size_t, int, int, int, int, const infinicore::DataType &, bool) const override;

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

private:
    size_t input_size_per_partition_;
    size_t output_size_per_partition_;
    bool is_k_full_;
};

} // namespace infinilm::quantization
