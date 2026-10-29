#include "../../layers/causal_lm_templates/text_causal_lm.hpp"
#include "../models_registry.hpp"
#include "../qwen3_5/qwen3_5_for_causal_lm.hpp"
#include "../qwen3_next/qwen3_next_allocate_kv_cache_tensors.hpp"
#include "clover_decoder_layer.hpp"
#include <base/clover_live_mask.h>

namespace infinilm::models::clover_moe {

using CloverModel = layers::causal_lm_templates::TextModel<CloverDecoderLayer>;

class CloverForCausalLM : public layers::causal_lm_templates::TextCausalLM<CloverModel> {
public:
    using Base = layers::causal_lm_templates::TextCausalLM<CloverModel>;
    using Base::Base;

    Output forward(const Input &input, global_state::ForwardContext &context) const override {
        auto &runtime = context.require_runtime();
        if (runtime.isGraphRecording()) {
            throw std::invalid_argument("Clover requires eager execution.");
        }
        context.diffusion_preview = input.diffusion_preview;
        context.diffusion_expert_budget = input.diffusion_expert_budget;
        context.attn_metadata.diffusion_preview = input.diffusion_preview;
        context.diffusion_block_size = model_config_->get<size_t>("block_size");
        if (!input.diffusion_preview) {
            denoise_step_ = 0;
        }
        if (input.diffusion_preview) {
            const auto tokens = input.input_ids.value()->numel();
            if (!input.sample_all_positions || tokens % context.diffusion_block_size) {
                throw std::invalid_argument("Clover preview requires complete blocks and all-position logits.");
            }
            context.diffusion_valid = runtime::Tensor::empty(runtime, {tokens}, runtime::DataType::kInt8);
            execution::submit<infini::ops::Fill>(runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()),
                                                 context.diffusion_valid, 1.0, context.diffusion_valid);
            if (input.diffusion_expert_budget) {
                bool refresh = denoise_step_ % 5 == 0 || !previous_budget_;
                const auto &ids = input.input_ids.value();
                if (!previous_ids_ || previous_ids_->shape() != ids->shape()) {
                    previous_ids_ = runtime::Tensor::empty(runtime, ids->shape(), ids->dtype());
                    refresh = true;
                }
                context.diffusion_live = runtime::Tensor::empty(runtime, {tokens}, runtime::DataType::kInt8);
                execution::submit<infini::ops::CloverLiveMask>(runtime, execution::RecordingPolicy::HostOnly,
                                                               infini::ops::CloverLiveMask::DefaultConfig(runtime.device().type()), ids, previous_ids_,
                                                               static_cast<int64_t>(model_config_->get<size_t>("mask_token_id")), refresh, context.diffusion_live);
            }
            previous_budget_ = input.diffusion_expert_budget;
            ++denoise_step_;
        }
        return Base::forward(input, context);
    }

    void reset_cache(const cache::CacheConfig *cache_config, global_state::ForwardContext &context,
                     backends::AttentionBackend backend, const engine::distributed::RankInfo &rank) override {
        if (backend != backends::AttentionBackend::STATIC_ATTN) {
            throw std::invalid_argument("Clover currently requires the static attention backend.");
        }
        auto &runtime = context.require_runtime();
        denoise_step_ = 0;
        previous_budget_ = false;
        previous_ids_.reset();
        auto allocated = qwen3_next::qwen3_next_allocate_cache_tensors(runtime, rank, cache_config, model_config_, backend);
        for (auto &state : allocated.ssm_state_tensors) {
            if (!state) {
                continue;
            }
            state = runtime::Tensor::empty(runtime, state->shape(), runtime::DataType::kFloat32);
            execution::submit<infini::ops::Fill>(runtime, infini::ops::Fill::DefaultConfig(runtime.device().type()), state, 0.0, state);
        }
        context.kv_cache_vec = std::move(allocated.kv_cache_tensors);
        context.conv_state_vec = std::move(allocated.conv_state_tensors);
        context.ssm_state_vec = std::move(allocated.ssm_state_tensors);
        cache_config_ = cache_config ? cache_config->unique_copy() : nullptr;
    }

private:
    mutable size_t denoise_step_{0};
    mutable bool previous_budget_{false};
    mutable runtime::Tensor previous_ids_;
};

std::shared_ptr<config::ModelConfig> create_clover_config(std::shared_ptr<config::ModelConfig> config) {
    if (config->get_dtype() != runtime::DataType::kBFloat16) {
        throw std::invalid_argument("Clover requires BF16 model weights.");
    }
    auto &json = config->get_config_json();
    if (json.at("model_type") != "clover_moe" || json.at("block_size") != 32 || json.at("linear_key_head_dim") != 128 || json.at("linear_value_head_dim") != 128 || json.at("linear_conv_kernel_dim") != 4 || json.at("num_experts") != 256 || json.at("num_experts_per_tok") != 8) {
        throw std::invalid_argument("Unsupported Clover architecture.");
    }
    json["norm_topk_prob"] = true;
    json["position_id_axes"] = 1;
    json["rope_parameters"]["mrope_section"] = {16, 8, 8};
    json["rope_parameters"]["mrope_interleaved"] = false;
    return qwen3_5::prepare_qwen3_5_model_config(config);
}

} // namespace infinilm::models::clover_moe

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(clover_moe,
                                           infinilm::models::clover_moe::CloverForCausalLM,
                                           infinilm::models::clover_moe::create_clover_config);
}
