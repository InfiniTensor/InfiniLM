#pragma once

#include "../../layers/common_modules.hpp"
#include "../../models/infinilm_model.hpp"
#include "gemma2_decoder_layer.hpp"
#include <memory>

namespace infinilm::models::gemma2 {

using Gemma2Model = infinilm::layers::causal_lm_templates::TextModel<Gemma2DecoderLayer>;

/**
 * @brief Gemma-2 causal LM.
 *
 * Modeled on TextCausalLM, with one addition: Gemma-2 applies final logit
 * soft-capping (`logits * (1/cap) * tanh(logits * cap)`) after the LM head.
 */
class Gemma2ForCausalLM : public infinilm::InfinilmModel {
public:
    Gemma2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    infinilm::InfinilmModel::Output forward(const infinilm::InfinilmModel::Input &input) const override;

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(Gemma2Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);

private:
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    float final_logit_softcapping_{0.0f};
    size_t pp_size_{1};
    size_t pp_stage_{0};
};

} // namespace infinilm::models::gemma2
