#pragma once

#include "../quantization/quantization.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/ops.hpp"
#include <infiniccl.h>
#include <optional>

namespace infinilm::nn {

using namespace infinicore::nn;

class BaseLinear : public infinicore::nn::Module {
public:
    BaseLinear(size_t in_features, size_t out_features,
               std::shared_ptr<infinilm::quantization::BaseQuantization> quantization = std::make_shared<infinilm::quantization::NoneQuantization>(nullptr),
               bool bias = true,
               const infinicore::DataType &dtype = infinicore::DataType::F32,
               const infinicore::Device &device = infinicore::Device(),
               int split_dim = -1, int tp_rank = 0, int tp_size = 1,
               int tp_num_heads = -1);

    // Forward pass: output = input @ weight.T + bias
    infinicore::Tensor forward(infinicore::Tensor &input) const;

    // Forward pass with residual connection
    infinicore::Tensor forward(infinicore::Tensor &input, infinicore::Tensor &residual) const;

    // Module information
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }
    infinicore::DataType dtype() const { return dtype_; }
    float alpha() const { return alpha_; }
    void set_alpha(float alpha) { alpha_ = alpha; }

    // Accessors for parameters (backward compatible)
    infinicore::Tensor weight() const;
    infinicore::Tensor bias() const;
    infinicore::Tensor weight_scale() const;
    infinicore::Tensor weight_zeros() const;
    infinicore::Tensor gidx() const;

    // Get parameter by name
    infinicore::Tensor get_param(const std::string &name) const;

    std::shared_ptr<infinilm::quantization::BaseQuantization> get_quantization() const { return quantization_; }
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

    // Split fused linear parameters into named sub-parameters
    std::vector<infinilm::quantization::SplitParam> split_params(
        const std::vector<infinilm::quantization::SplitInfo> &splits,
        int tp_rank, int tp_size, int tp_num_heads) const;

    // Allow subclasses to access the raw parameters map
    const infinicore::nn::Parameter &get_parameter_ref(const std::string &name) const;

protected:
    infinicore::Tensor compute_linear(infinicore::Tensor &input) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    infinicore::DataType dtype_;
    int split_dim_ = -1;
    float alpha_ = 1.0f;
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_;
};

} // namespace infinilm::nn
