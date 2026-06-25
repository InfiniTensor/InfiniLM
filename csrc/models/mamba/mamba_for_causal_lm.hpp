#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/linear/linear.hpp"
#include "../infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <memory>
#include <vector>

namespace infinilm::models::mamba {

class MambaMixer : public infinicore::nn::Module {
public:
    MambaMixer(std::shared_ptr<infinilm::config::ModelConfig> config,
               size_t layer_idx,
               const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t layer_idx_;
    size_t intermediate_size_;
    size_t state_size_;
    size_t time_step_rank_;
    size_t conv_kernel_;
    infinicore::Tensor dt_bias_zero_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, in_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, x_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, dt_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, out_proj);
    INFINICORE_NN_PARAMETER(conv1d_weight);
    INFINICORE_NN_PARAMETER(conv1d_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_PARAMETER(D);
};

class MambaBlock : public infinicore::nn::Module {
public:
    MambaBlock(std::shared_ptr<infinilm::config::ModelConfig> config,
               size_t layer_idx,
               const infinicore::Device &device);
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(MambaMixer, mixer);
};

class MambaModel : public infinicore::nn::Module {
public:
    MambaModel(std::shared_ptr<infinilm::config::ModelConfig> config,
               const infinicore::Device &device);
    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

private:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embeddings);
    INFINICORE_NN_MODULE_VEC(MambaBlock, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm_f);
};

class MambaForCausalLM : public infinilm::InfinilmModel {
public:
    MambaForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                     const infinicore::Device &device);
    Output forward(const Input &input) const override;
    void reset_cache(const cache::CacheConfig *cache_config) override;

private:
    INFINICORE_NN_MODULE(MambaModel, backbone);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

std::shared_ptr<infinilm::config::ModelConfig>
create_mamba_model_config(std::shared_ptr<infinilm::config::ModelConfig> config);

} // namespace infinilm::models::mamba
