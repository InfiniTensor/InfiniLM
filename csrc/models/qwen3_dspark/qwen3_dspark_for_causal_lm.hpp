#pragma once

#include "../../layers/causal_lm_templates/text_decoder_layer.hpp"
#include "../../layers/common_modules.hpp"
#include "qwen3_dspark_attention.hpp"

namespace infinilm::models::qwen3_dspark {

using Qwen3DSparkDecoderBase = infinilm::layers::causal_lm_templates::TextDecoderLayer<
    Qwen3DSparkAttention, infinilm::layers::MLP>;

class Qwen3DSparkDecoderLayer final : public Qwen3DSparkDecoderBase {
public:
    using Qwen3DSparkDecoderBase::Qwen3DSparkDecoderBase;

    void inject_target_hidden(
        const infinilm::runtime::Tensor &positions,
        const infinilm::runtime::Tensor &hidden_states,
        const infinilm::runtime::Tensor &slot_mapping,
        infinilm::global_state::ForwardContext &forward_context) const {
        self_attn_->inject_target_hidden(
            positions, hidden_states, slot_mapping, forward_context);
    }
};

class Qwen3DSparkForCausalLM final : public InfinilmModel {
public:
    Qwen3DSparkForCausalLM(
        infinilm::runtime::Runtime &runtime,
        std::shared_ptr<config::ModelConfig> model_config,
        const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend,
        const layers::moe::EPConfig &ep_config);

    Output forward(
        const Input &input,
        infinilm::global_state::ForwardContext &forward_context) const override;

    bool supports_kv_only_prepare() const override { return true; }

private:
    Output forward_markov(
        const Input &input,
        infinilm::global_state::ForwardContext &forward_context) const;
    Output apply_markov(
        const infinilm::runtime::Tensor &base_logits,
        const infinilm::runtime::Tensor &previous_tokens,
        infinilm::runtime::Runtime &runtime) const;
    void inject_target_hidden(
        const Input &input,
        infinilm::global_state::ForwardContext &forward_context) const;
    infinilm::runtime::Tensor dummy_logits(
        const Input &input,
        infinilm::runtime::Runtime &runtime) const;

    size_t hidden_size_{0};
    size_t vocab_size_{0};
    size_t markov_rank_{0};
    std::shared_ptr<infinilm::layers::Embedding> embed_tokens_;
    std::vector<std::shared_ptr<Qwen3DSparkDecoderLayer>> layers_;
    std::shared_ptr<infinilm::layers::RMSNorm> norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> fc_;
    std::shared_ptr<infinilm::layers::RMSNorm> hidden_norm_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> lm_head_;
    std::shared_ptr<infinilm::layers::Embedding> markov_w1_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> markov_w2_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> confidence_proj_;
};

std::shared_ptr<infinilm::config::ModelConfig>
create_qwen3_dspark_model_config(
    std::shared_ptr<infinilm::config::ModelConfig> model_config);

} // namespace infinilm::models::qwen3_dspark
