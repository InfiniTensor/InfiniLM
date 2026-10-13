#pragma once
#include "../../weights/weights.hpp"

#include "base_quantization.hpp"
#include <optional>
namespace infinilm::quantization {

class NoneQuantization : public BaseQuantization {
public:
    explicit NoneQuantization(const nlohmann::json &quant_config,
                              std::optional<infinilm::runtime::DataType> output_dtype = std::nullopt)
        : BaseQuantization(quant_config), output_dtype_(output_dtype) {};

    NoneQuantization();

    const std::optional<infinilm::runtime::DataType> &output_dtype() const { return output_dtype_; }

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::NONE;
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

    // Ascend: pre-pack weight to [IC, OC] after loading to skip runtime permute.
    // Returns shared_from_this() only on Ascend; nullptr otherwise (no-op).
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        infinilm::runtime::Runtime &runtime, ParamsMap &params, bool pre_transpose,
        int split_dim = -1) const override;

private:
    const std::optional<infinilm::runtime::DataType> output_dtype_;
    mutable bool weight_prepacked_ = false; // true when weight was pre-packed for Ascend
};

} // namespace infinilm::quantization
