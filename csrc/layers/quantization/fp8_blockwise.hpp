#pragma once

#include "base_quantization.hpp"

namespace infinilm::quantization {

// FP8 (E4M3) weights with block-wise F32 scales (DeepSeek-V3 / Qwen3-FP8
// style, quant_method == "fp8"). Weight is stored as [out, in] F8 with one
// scale per [BM, BN] block in weight_scale_inv [out / BM, in / BN] F32.
class FP8Blockwise : public BaseQuantization {
public:
    explicit FP8Blockwise(const nlohmann::json &quant_config);

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::FP8_W8A16;
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
    // Block shape parsed from "weight_block_size" (default [128, 128]).
    // block_m_ divides the output (dim0), block_n_ the input (dim1) extent.
    size_t block_m_ = 128;
    size_t block_n_ = 128;
};

} // namespace infinilm::quantization
