#pragma once

#include "../infinilm_model.hpp"
#include "llama_model.hpp"

#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::llama {

/**
 * @brief Llama model for Causal Language Modeling
 *
 * Extends LlamaModel by adding a language modeling head (lm_head) that
 * projects hidden states to vocabulary logits.
 *
 * This matches the structure of HuggingFace's LlamaForCausalLM.
 */
class LlamaForCausalLM : public InfinilmModel {
public:
    /**
     * @brief Construct LlamaForCausalLM module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     */
    LlamaForCausalLM(const LlamaConfig &config,
                     const infinicore::Device &device,
                     engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: compute language modeling logits
     *
     * @param input Encapsulated input tensors and other parameters
     * @return Output structure containing the result
     */
    Output forward(const Input &input) const;

    void reset_cache(const cache::CacheConfig *cache_config) override;

    // Module information
    const LlamaConfig &config() const { return model_->config(); }
    LlamaModel &model() { return *model_; }
    const LlamaModel &model() const { return *model_; }

protected:
    // Base model
    INFINICORE_NN_MODULE(LlamaModel, model);

    // Language modeling head
    INFINICORE_NN_MODULE(infinicore::nn::Linear, lm_head);
};

} // namespace infinilm::models::llama
