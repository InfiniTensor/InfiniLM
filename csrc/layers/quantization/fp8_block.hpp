#pragma once

#include "none_quantization.hpp"

namespace infinilm::quantization {

// Block-scaled E4M3 checkpoints with a portable W8A16 compatibility path and
// opt-in NVIDIA Marlin packing during the existing post-load lifecycle.
class FP8Block : public NoneQuantization {
public:
    explicit FP8Block(const nlohmann::json &config);

    QuantScheme get_quant_scheme() const override { return QuantScheme::FP8_BLOCK_W8A16; }
    std::vector<ParamDescriptor> get_param_layout(
        size_t in_features, size_t out_features, int split_dim, int tp_rank,
        int tp_size, int tp_num_heads, const infinicore::DataType &dtype, bool bias) const override;
    std::vector<SplitParam> split_params(
        const std::unordered_map<std::string, infinicore::nn::Parameter> &params,
        const std::vector<SplitInfo> &splits, int narrow_dim,
        int tp_rank, int tp_size, int tp_num_heads) const override;
    infinicore::Tensor forward(const ParamsMap &params, const infinicore::Tensor &input,
                               bool has_bias, float alpha = 1.0f) const override;
    infinicore::Tensor forward_allreduce(const ParamsMap &params, const infinicore::Tensor &input,
                                         bool has_bias, infinicclComm_t communicator,
                                         float alpha = 1.0f) const override;
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        ParamsMap &, const infinicore::Device &, int = -1) const override;

private:
    static constexpr size_t block_size_ = 128;
    mutable infinicore::DataType activation_dtype_ = infinicore::DataType::BF16;
};

} // namespace infinilm::quantization
