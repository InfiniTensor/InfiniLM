#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../linear/linear.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/distributed/allgather.hpp"
#include "infinicore/ops/select_last_token_hidden_states.hpp"

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
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        pp_size_ = static_cast<size_t>(rank_info.pp_size);
        pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
        tp_size_ = static_cast<size_t>(rank_info.tp_size);
        tp_rank_ = static_cast<size_t>(rank_info.tp_rank);
        vocab_parallel_ = device.getType() == infinicore::Device::Type::HYGON
            && tp_size_ > 1
            && vocab_size % tp_size_ == 0;

        model_ = this->register_module<Model>("model", model_config, device);
        if (is_last_pp_stage()) {
            lm_head_ = this->register_module<infinilm::layers::linear::ColumnParallelLinear>(
                "lm_head",
                hidden_size,
                vocab_size,
                false,
                dtype,
                device,
                vocab_parallel_ ? tp_rank_ : 0,
                vocab_parallel_ ? tp_size_ : 1);
        }
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto hidden_states = model_->forward(input);
        if (!is_last_pp_stage()) {
            return {infinicore::Tensor(), hidden_states};
        }

        if (input.last_token_only) {
            if (!input.input_offsets.has_value()) {
                throw std::runtime_error("TextCausalLM: last_token_only requires input_offsets");
            }
            hidden_states = infinicore::op::select_last_token_hidden_states(
                hidden_states, input.input_offsets.value());
        }

        auto logits = gather_logits(lm_head_->forward(hidden_states));
        return {logits, hidden_states};
    }

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const {
        if (!lm_head_) {
            throw std::runtime_error("TextCausalLM::logits_from_hidden called on a non-last pipeline stage");
        }
        return gather_logits(
            lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states)));
    }

    Model &model() { return *model_; }

protected:
    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, lm_head);

private:
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    infinicore::Tensor gather_logits(const infinicore::Tensor &local_logits) const {
        if (!vocab_parallel_) {
            return local_logits;
        }

        const auto &local_shape = local_logits->shape();
        if (local_shape.empty() || local_shape.back() == 0) {
            throw std::runtime_error("TextCausalLM: invalid local logits shape");
        }
        const size_t local_vocab_size = local_shape.back();
        const size_t num_rows = local_logits->numel() / local_vocab_size;
        auto local_flat = local_logits->view({num_rows, local_vocab_size});
        const auto &rank_info =
            infinilm::global_state::get_tensor_model_parallel_rank_info();
        auto gathered = infinicore::op::distributed::allgather(
            local_flat, tp_size_, rank_info.comm);

        auto output_shape = local_shape;
        output_shape.back() *= tp_size_;
        return gathered->view({tp_size_, num_rows, local_vocab_size})
            ->permute({1, 0, 2})
            ->contiguous()
            ->view(output_shape);
    }

    size_t pp_size_{1};
    size_t pp_stage_{0};
    size_t tp_size_{1};
    size_t tp_rank_{0};
    bool vocab_parallel_{false};
};

} // namespace infinilm::layers::causal_lm_templates
