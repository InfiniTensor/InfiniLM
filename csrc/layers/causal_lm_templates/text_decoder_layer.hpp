#include "../../models/models_registry.hpp"
#include <exception>
#pragma once

#include "../../config/model_config.hpp"
#include "../../global_state/forward_context.hpp"
#include "../../weights/weights.hpp"
#include "../normalization/rmsnorm.hpp"
#include "runtime/device.hpp"
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Generic Text decoder layer (transformer block) class.
 *
 * @tparam Attention: The attention module type (e.g., Qwen3Attention)
 * @tparam MLP: The MLP module type (e.g., Qwen3MLP)
 */
template <typename Attention, typename MLP>
class TextDecoderLayer : public infinilm::weights::ModelWeights {
public:
    TextDecoderLayer(infinilm::runtime::Runtime &runtime, std::shared_ptr<config::ModelConfig> model_config,
                     size_t layer_idx, const engine::distributed::RankInfo &rank_info, backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config) : layer_idx_(layer_idx) {
        models::validate_runtime_construction(model_config, runtime, rank_info);
        try {

            const auto &dtype{model_config->get_dtype()};
            size_t hidden_size = model_config->get<size_t>("hidden_size");
            double rms_norm_eps = model_config->get<double>("rms_norm_eps");

            input_layernorm_ = this->bind_child("input_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype));
            post_attention_layernorm_ = this->bind_child("post_attention_layernorm", std::make_shared<infinilm::layers::RMSNorm>(runtime, hidden_size, rms_norm_eps, dtype));

            self_attn_ = this->bind_child("self_attn", std::make_shared<Attention>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
            static_assert(std::is_constructible_v<MLP, infinilm::runtime::Runtime &, std::shared_ptr<config::ModelConfig>, size_t, const engine::distributed::RankInfo &, backends::AttentionBackend, const layers::moe::EPConfig &>, "selected MLP requires explicit constructor, no Device fallback");
            mlp_ = this->bind_child("mlp", std::make_shared<MLP>(runtime, model_config, layer_idx, rank_info, attention_backend, ep_config));
            if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
                runtime.syncStream();
            }
        } catch (...) {
            if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
                try {
                    runtime.syncStream();
                } catch (...) { std::terminate(); }
            }
            throw;
        }
    }

    std::tuple<infinilm::runtime::Tensor, infinilm::runtime::Tensor> forward(
        const infinilm::runtime::Tensor &positions,
        infinilm::runtime::Tensor &hidden_states,
        infinilm::runtime::Tensor &residual,
        infinilm::global_state::ForwardContext &forward_context) {
        if (forward_context.runtime == nullptr) {
            throw std::invalid_argument("decoder forward requires an explicit Runtime");
        }
        input_layernorm_->forward_inplace(*forward_context.runtime, hidden_states, residual);
        hidden_states = forward_attention(positions, hidden_states, forward_context);
        post_attention_layernorm_->forward_inplace(*forward_context.runtime, hidden_states, residual);
        hidden_states = mlp_->forward(*forward_context.runtime, hidden_states);
        return std::make_tuple(hidden_states, residual);
    }

    size_t layer_idx() const { return layer_idx_; }

protected:
    std::shared_ptr<infinilm::layers::RMSNorm> input_layernorm_;
    std::shared_ptr<infinilm::layers::RMSNorm> post_attention_layernorm_;
    std::shared_ptr<Attention> self_attn_;
    std::shared_ptr<MLP> mlp_;

    size_t layer_idx_;

private:
    infinilm::runtime::Tensor forward_attention(
        const infinilm::runtime::Tensor &positions,
        const infinilm::runtime::Tensor &hidden_states,
        infinilm::global_state::ForwardContext &forward_context) {
        return self_attn_->forward(positions, hidden_states, forward_context);
    }
};

} // namespace infinilm::layers::causal_lm_templates
