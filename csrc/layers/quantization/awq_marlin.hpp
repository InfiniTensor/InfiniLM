#pragma once

#include "base_quantization.hpp"

namespace infinilm::quantization {

class AWQMarlin : public BaseQuantization {
public:
    AWQMarlin(const nlohmann::json &quant_config, size_t input_size_per_partition, size_t output_size_per_partition)
        : BaseQuantization(quant_config),
          input_size_per_partition_(input_size_per_partition),
          output_size_per_partition_(output_size_per_partition) {}

    QuantScheme get_quant_scheme() const override { return QuantScheme::AWQ_MARLIN_W4A16; }

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

    void reset_runtime_state() const override;

private:
    infinicore::Tensor get_workspace(
        infinicore::Tensor out,
        const infinicore::Tensor &a,
        const infinicore::Tensor &b,
        infinicore::Tensor &b_scales,
        infinicore::Tensor &global_scales,
        infinicore::Tensor &b_zeros,
        infinicore::Tensor &g_idx,
        infinicore::Tensor &perm) const;

    size_t input_size_per_partition_;
    size_t output_size_per_partition_;
    // Per-layer Marlin workspace. It must be all-zero before each launch
    // because the current InfiniCore Marlin kernels use it as lock state.
    // TODO: replace per-layer memset with a shared global zero workspace, or
    // update the kernels so the lock region is self-reset at completion. The
    // remaining gap to vLLM is mainly from two sources: TP communication cost
    // and this workspace memset/reset path.
    mutable infinicore::Tensor workspace_;
};

} // namespace infinilm::quantization
