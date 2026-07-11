#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../linear/linear.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include <infiniccl.h>
#include <stdexcept>

namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Text Causal Language Modeling class
 *
 * A generic template class for Causal Language Modeling.
 *
 * @tparam Model The base model type (e.g., Qwen3Model, Qwen3MoeModel)
 *
 * Usage example:
 * @code
 * using Qwen3CausalLM = TextCausalLM<Qwen3Model>;
 * @endcode
 */
template <typename Model>
class TextCausalLM : public InfinilmModel {
public:
    /**
     * @brief Construct TextCausalLM module
     *
     * @param model_config: Model configuration.
     * @param device: Device to create tensors on
     */
    TextCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device) {
        model_config_ = model_config;

        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t vocab_size = model_config->get<size_t>("vocab_size");
        const auto &dtype{model_config->get_dtype()};

        model_ = this->register_module<Model>("model", model_config, device);
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        tp_size_ = static_cast<size_t>(rank_info.tp_size);
        tp_communicator_ = rank_info.comm;
        vocab_parallel_ = tp_size_ > 1
                       && model_config->get<std::string>("model_type") == "qwen3_moe"
                       && vocab_size % tp_size_ == 0
                       && model_config->get_quant_scheme() == infinilm::quantization::QuantScheme::NONE;
        if (vocab_parallel_) {
            lm_head_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
                "lm_head", hidden_size, vocab_size, false, dtype, device,
                rank_info.tp_rank, rank_info.tp_size);
        } else {
            lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>(
                "lm_head", hidden_size, vocab_size, false, dtype, device);
        }
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto hidden_states = model_->forward(input);
        auto lm_head_input = hidden_states;
        // Ordinary prefill needs only the final-position logits.
        if (!input.sample_all_positions
            && input.last_token_indices.has_value()
            && !input.last_token_indices->empty()
            && hidden_states->size(0) == 1
            && hidden_states->size(1) > input.last_token_indices->size()) {
            const auto &indices = input.last_token_indices.value();
            for (const size_t index : indices) {
                if (index >= hidden_states->size(1)) {
                    throw std::out_of_range("last-token index exceeds packed hidden states");
                }
            }
            if (indices.size() == 1) {
                lm_head_input = hidden_states->narrow({{1, indices.front(), 1}});
            } else {
                std::vector<infinicore::Tensor> final_hidden_states;
                final_hidden_states.reserve(indices.size());
                for (const size_t index : indices) {
                    final_hidden_states.push_back(
                        hidden_states->narrow({{1, index, 1}}));
                }
                lm_head_input = infinicore::op::cat(std::move(final_hidden_states), 1);
            }
        }
        auto logits = lm_head_->forward(lm_head_input);
        if (!input.use_local_vocab_logits) {
            logits = gather_vocab_logits_(std::move(logits));
        }
        return {logits};
    }

    /// Return full-vocabulary logits, gathering rank-local shards when needed.
    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const {
        return gather_vocab_logits_(
            lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states)));
    }

    Model &model() { return *model_; }

    bool uses_vocab_parallel_logits() const override {
        return vocab_parallel_;
    }

protected:
    INFINICORE_NN_MODULE(Model, model);
    infinicore::Tensor gather_vocab_logits_(infinicore::Tensor logits) const {
        if (!vocab_parallel_) {
            return logits;
        }
        auto output_shape = logits->shape();
        const size_t local_vocab_size = output_shape.back();
        const size_t positions = logits->numel() / local_vocab_size;
        auto gathered = infinicore::op::distributed::allgather(
            logits, tp_size_, tp_communicator_);
        output_shape.back() = local_vocab_size * tp_size_;
        if (positions == 1) {
            return gathered->view(output_shape);
        }
        return gathered->view({tp_size_, positions, local_vocab_size})
            ->permute({1, 0, 2})
            ->contiguous()
            ->view(output_shape);
    }

    // The base type holds either a replicated or column-parallel registered head.
    INFINICORE_NN_MODULE(infinilm::layers::linear::BaseLinear, lm_head);
    bool vocab_parallel_{false};
    size_t tp_size_{1};
    infinicclComm_t tp_communicator_{nullptr};
};

} // namespace infinilm::layers::causal_lm_templates
