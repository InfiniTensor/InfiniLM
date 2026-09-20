#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace infinilm::models::mamba2 {

struct Mamba2BatchMetadata {
    std::vector<int32_t> input_offsets;
    std::vector<int32_t> init_state_indices;
    std::vector<int32_t> final_state_indices;
};

class Mamba2Mixer : public infinicore::nn::Module {
public:
    Mamba2Mixer(std::shared_ptr<infinilm::config::ModelConfig> config,
                size_t layer_idx,
                const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const Mamba2BatchMetadata &metadata) const;

private:
    size_t layer_idx_;
    size_t hidden_size_;
    size_t intermediate_size_;
    size_t state_size_;
    size_t num_heads_;
    size_t head_dim_;
    size_t num_groups_;
    size_t conv_dim_;
    size_t conv_kernel_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, in_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, out_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_PARAMETER(conv1d_weight);
    INFINICORE_NN_PARAMETER(conv1d_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_PARAMETER(D);
    INFINICORE_NN_PARAMETER(dt_bias);
};

class Mamba2Block : public infinicore::nn::Module {
public:
    Mamba2Block(std::shared_ptr<infinilm::config::ModelConfig> config,
                size_t layer_idx,
                const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const Mamba2BatchMetadata &metadata) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(Mamba2Mixer, mixer);
};

class Mamba2Model : public infinicore::nn::Module {
public:
    Mamba2Model(std::shared_ptr<infinilm::config::ModelConfig> config,
                const infinicore::Device &device);
    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

private:
    static Mamba2BatchMetadata build_batch_metadata_(
        const infinilm::InfinilmModel::Input &input);
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embedding);
    INFINICORE_NN_MODULE_VEC(Mamba2Block, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm_f);
};

class Mamba2ForCausalLM
    : public infinilm::layers::causal_lm_templates::TextCausalLM<Mamba2Model> {
public:
    Mamba2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                      const infinicore::Device &device);
    void reset_cache(const cache::CacheConfig *cache_config) override;
    bool supports_graph_compilation() const override { return false; }
};

std::shared_ptr<infinilm::config::ModelConfig>
create_mamba2_model_config(std::shared_ptr<infinilm::config::ModelConfig> config);

} // namespace infinilm::models::mamba2
