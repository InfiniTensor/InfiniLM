#pragma once

#include "../quantization/quantization.hpp"
#include "../quantization/w4_execution.hpp"
#include "../../weights/weights.hpp"
#include <infiniccl/infiniccl.h>
#include <optional>

namespace infinilm::nn {

using infinilm::weights::Weight;

class BaseLinear : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    // Runtime-first supply.
    BaseLinear(infinilm::runtime::Runtime &runtime, size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
               bool bias = true,
               const infinilm::runtime::DataType &dtype = infinilm::runtime::DataType::kFloat32,
               int split_dim = -1, int tp_rank = 0, int tp_size = 1,
               int tp_num_heads = -1);

    // Forward pass: output = input @ weight.T + bias
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const;

    // Forward pass with residual connection
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input,
                              infinilm::runtime::Tensor &residual) const;

    // Module information
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }
    infinilm::runtime::DataType dtype() const { return dtype_; }
    infinilm::runtime::DataType output_dtype() const { return output_dtype_; }
    float alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }

    // Accessors for parameters (backward compatible)
    infinilm::runtime::Tensor weight() const;
    infinilm::runtime::Tensor bias() const;

    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization() const { return quantization_; }
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;
    void validate_awq_metadata(infinilm::runtime::Runtime &runtime) const;
    // Pure checkpoint metadata + constructor geometry, never device-value readback.
    std::optional<int64_t> validate_gptq_metadata(
        infinilm::runtime::Runtime &runtime, bool retained_global_scale_table = false) const;
    // Loading-only stage; raw Parameter registrations and published binding stay
    // unchanged until publish. Caller must complete raw TP/fused loading first.
    std::shared_ptr<const infinilm::quantization::W4Execution> prepare_w4_after_loading(
        infinilm::runtime::Runtime &runtime, bool retained_global_scale_table = false);
    // Precondition: nonnull candidate returned by this layer's successful prepare;
    // model/worker must be quiescent.
    void publish_w4_after_loading(std::shared_ptr<const infinilm::quantization::W4Execution> candidate) noexcept;

    // Split fused linear parameters into named sub-parameters
    std::vector<infinilm::quantization::SplitParam> split_params(
        const std::vector<infinilm::quantization::SplitInfo> &splits,
        int tp_rank, int tp_size, int tp_num_heads) const;

    // Allow subclasses to access the raw parameters map
    const infinilm::weights::Weight &get_parameter_ref(const std::string &name) const;

protected:
    infinilm::runtime::Tensor compute_linear(infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input) const;
    infinilm::runtime::Tensor compute_linear_allreduce(
        infinilm::runtime::Runtime &runtime, infinilm::runtime::Tensor &input, infinicclComm_t communicator) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    infinilm::runtime::DataType dtype_;
    const infinilm::runtime::DataType output_dtype_;
    int split_dim_ = -1;
    const int loading_tp_rank_;
    const int loading_tp_size_;
    float alpha_ = 1.0f;
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_;
    // Bound once at construction and refreshed after post-load repacking.
    // The Tensor handles share the parameter storage while weights are loaded.
    infinilm::quantization::ParamsMap bound_params_;
    std::shared_ptr<const infinilm::quantization::W4Execution> w4_execution_;
    std::weak_ptr<const infinilm::quantization::W4Execution> w4_staged_identity_;
    bool w4_prepared_ = false; // successful prepare is single-use, not a model Ready state
    // Typed loading is one attempt, including a successful no-op. This is not
    // Ready: a failed transformation may have mutated shared quantization state.
    bool typed_postload_started_ = false;
};

} // namespace infinilm::nn
