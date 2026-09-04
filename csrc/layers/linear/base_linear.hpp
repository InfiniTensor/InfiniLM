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
               int tp_num_heads = -1,
               const std::string &stem = "");

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

    // One shard of a fused linear, for schemes that cannot share a single fused
    // buffer (GGUF block quantization: row_bytes differs per shard type).
    struct FusedShard {
        std::string name;    // Name used when registering with the parent module.
        size_t out_features; // Logical output rows for this shard.
        std::string stem;    // Checkpoint stem used for quantization lookup.
    };

    // Allocate one buffer per fused Linear shard. Local parameter keys use
    // "shard<i>.<suffix>", while returned names use "<name>.<suffix>" for
    // registration with the parent module. This path is selected when the
    // quantization scheme returns an empty layout for the fused group.
    // shard_stems_ preserves each checkpoint stem for per-shard lookup during
    // forward execution.
    std::vector<infinilm::quantization::SplitParam> init_fused_shards(
        const std::vector<FusedShard> &shards);

    // Allow subclasses to access the raw parameters map
    const infinicore::nn::Parameter &get_parameter_ref(const std::string &name) const;

protected:
    infinicore::Tensor compute_linear(infinicore::Tensor &input) const;
    infinicore::Tensor compute_linear_allreduce(
        infinicore::Tensor &input, infinicclComm_t communicator) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
    infinicore::DataType dtype_;
    int split_dim_ = -1;
    float alpha_ = 1.0f;
    std::string stem_; // Checkpoint tensor path used by name-based quantization lookup.
    // Per-shard checkpoint stems recorded by init_fused_shards. The index
    // matches the i in the corresponding "shard<i>.*" parameter key.
    // This vector is empty for non-fused Linear layers.
    std::vector<std::string> shard_stems_;
    bool sharded_ = false; // Fused layout whose parameters live in shard<i>.* buffers.
    std::shared_ptr<infinilm::quantization::BaseQuantization> quantization_;
};

} // namespace infinilm::nn
