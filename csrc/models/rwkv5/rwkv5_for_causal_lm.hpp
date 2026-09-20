#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/layer_norm.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace infinilm::models::rwkv5 {

struct RWKV5BatchMetadata {
    std::vector<int32_t> input_offsets;
    std::vector<int32_t> init_state_indices;
    std::vector<int32_t> final_state_indices;
};

class RWKV5HeadGroupNorm : public infinicore::nn::Module {
public:
    RWKV5HeadGroupNorm(size_t hidden_size,
                       size_t num_heads,
                       double eps,
                       const infinicore::DataType &dtype,
                       const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t num_heads_;
    size_t head_size_;
    double eps_;
    infinicore::Tensor unit_weight_;
    infinicore::Tensor zero_bias_;
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
};

class RWKV5TimeMix : public infinicore::nn::Module {
public:
    RWKV5TimeMix(std::shared_ptr<infinilm::config::ModelConfig> config,
                 size_t layer_idx,
                 const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const RWKV5BatchMetadata &metadata) const;

private:
    infinicore::Tensor run_wkv_(const infinicore::Tensor &receptance,
                               const infinicore::Tensor &key,
                               const infinicore::Tensor &value,
                               const RWKV5BatchMetadata &metadata) const;

    size_t layer_idx_;
    size_t hidden_size_;
    size_t num_heads_;
    size_t head_size_;
    bool use_gate_;

    INFINICORE_NN_PARAMETER(time_mix_k);
    INFINICORE_NN_PARAMETER(time_mix_v);
    INFINICORE_NN_PARAMETER(time_mix_r);
    INFINICORE_NN_PARAMETER(time_mix_g);
    INFINICORE_NN_PARAMETER(time_decay);
    INFINICORE_NN_PARAMETER(time_faaaa);

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, key);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, value);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, receptance);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, gate);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, output);
    INFINICORE_NN_MODULE(RWKV5HeadGroupNorm, ln_x);
};

class RWKV5ChannelMix : public infinicore::nn::Module {
public:
    RWKV5ChannelMix(std::shared_ptr<infinilm::config::ModelConfig> config,
                    size_t layer_idx,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const RWKV5BatchMetadata &metadata) const;

private:
    size_t layer_idx_;
    INFINICORE_NN_PARAMETER(time_mix_k);
    INFINICORE_NN_PARAMETER(time_mix_r);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, key);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, value);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, receptance);
};

class RWKV5Block : public infinicore::nn::Module {
public:
    RWKV5Block(std::shared_ptr<infinilm::config::ModelConfig> config,
               size_t layer_idx,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const RWKV5BatchMetadata &metadata) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln1);
    INFINICORE_NN_MODULE(RWKV5TimeMix, att);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln2);
    INFINICORE_NN_MODULE(RWKV5ChannelMix, ffn);
};

class RWKV5Model : public infinicore::nn::Module {
public:
    RWKV5Model(std::shared_ptr<infinilm::config::ModelConfig> config,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

private:
    static RWKV5BatchMetadata build_batch_metadata_(
        const infinilm::InfinilmModel::Input &input);

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embeddings);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln0);
    INFINICORE_NN_MODULE_VEC(RWKV5Block, blocks);
    INFINICORE_NN_MODULE(infinicore::nn::LayerNorm, ln_out);
};

class RWKV5ForCausalLM
    : public infinilm::layers::causal_lm_templates::TextCausalLM<RWKV5Model> {
public:
    RWKV5ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                     const infinicore::Device &device);

    void reset_cache(const cache::CacheConfig *cache_config) override;
    bool supports_graph_compilation() const override { return false; }
};

std::shared_ptr<infinilm::config::ModelConfig>
create_rwkv5_model_config(std::shared_ptr<infinilm::config::ModelConfig> config);

} // namespace infinilm::models::rwkv5
