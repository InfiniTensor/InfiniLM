#pragma once

#include "../llama/llama_for_causal_lm.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <stdexcept>

namespace infinilm::models::llama {

/**
 * @brief Wrapper class for LlamaForCausalLM that provides the same interface
 *
 * This wrapper encapsulates a LlamaForCausalLM object and forwards all
 * method calls to the underlying model. It's designed to be a drop-in
 * replacement for LlamaForCausalLM in the Python bindings.
 */
class LlamaForCausalLMWrapper {
public:
    /**
     * @brief Construct wrapper with underlying LlamaForCausalLM
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     * @param dtype Data type for model parameters
     */
    LlamaForCausalLMWrapper(const LlamaConfig &config,
                            const infinicore::Device &device,
                            infinicore::DataType dtype = infinicore::DataType::F32);

    /**
     * @brief Forward pass: compute language modeling logits
     *
     * @param input_ids Token IDs tensor of shape [batch, seq_len]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_caches Optional KV caches for incremental decoding (one per layer)
     * @return Logits tensor of shape [batch, seq_len, vocab_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids,
                               std::vector<void *> *kv_caches = nullptr) const;

    /**
     * @brief Get model configuration
     *
     * @return Reference to model configuration
     */
    const LlamaConfig &config() const;

    /**
     * @brief Get the underlying LlamaForCausalLM model
     *
     * @return Reference to wrapped model
     */
    LlamaForCausalLM &model();

    /**
     * @brief Get the underlying LlamaForCausalLM model (const version)
     *
     * @return Const reference to wrapped model
     */
    const LlamaForCausalLM &model() const;

    /**
     * @brief Get model state dictionary
     *
     * @return Map from parameter names to Parameter objects
     */
    std::unordered_map<std::string, infinicore::nn::Parameter> state_dict() const;

    /**
     * @brief Get a specific parameter by name
     *
     * @param name Parameter name
     * @return Parameter object
     * @throws std::runtime_error if parameter not found
     */
    infinicore::nn::Parameter get_parameter(const std::string &name) const;

    /**
     * @brief Load state dictionary into model
     *
     * @param state_dict Map from parameter names to tensors
     */
    void load_state_dict(const std::unordered_map<std::string, infinicore::Tensor> &state_dict);

private:
    std::shared_ptr<LlamaForCausalLM> model_;
};

} // namespace infinilm::models::llama
