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

    // With pre_transpose, materialize [IC, OC]. Eligible Ascend weights are
    // additionally converted to the ACLNN MatmulWeightNz physical layout.
    std::shared_ptr<BaseQuantization> process_weights_after_loading(
        infinilm::runtime::Runtime &runtime, ParamsMap &params, bool pre_transpose,
        int split_dim = -1) const override;

private:
    enum class WeightLayout {
        None,
        PackedNd,
        AscendNz,
    };

    infinilm::runtime::Tensor narrow_nz_output(
        infinilm::runtime::Runtime &runtime,
        const infinilm::runtime::Tensor &output,
        const infinilm::runtime::Tensor &weight) const;

    const std::optional<infinilm::runtime::DataType> output_dtype_;
    mutable WeightLayout weight_layout_ = WeightLayout::None;
    mutable size_t nz_output_width_ = 0;
};

} // namespace infinilm::quantization
