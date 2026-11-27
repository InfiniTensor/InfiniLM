#include "gemma4_for_causal_lm.hpp"

#include "../../execution/submit.hpp"
#include "../../parallel/topology.hpp"
#include "../models_registry.hpp"
#include "gemma4_allocate_kv_cache_tensors.hpp"
#include "gemma4_ops.hpp"
#include <base/add.h>
#include <base/embedding.h>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>

namespace infinilm::models::gemma4 {
namespace {

std::vector<int64_t> read_int_vector_host(infinilm::runtime::Runtime &runtime,
                                          const infinilm::runtime::Tensor &tensor) {
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto read_bytes = [&](void *dst) {
        if (tensor->device() == cpu) {
            std::memcpy(dst, tensor->data(), tensor->nbytes());
        } else {
            runtime.memcpyD2H(dst, tensor->data(), tensor->nbytes());
            runtime.syncStream();
        }
    };
    std::vector<int64_t> out(tensor->numel());
    if (tensor->dtype() == infinilm::runtime::DataType::kInt64) {
        read_bytes(out.data());
    } else if (tensor->dtype() == infinilm::runtime::DataType::kInt32) {
        std::vector<int32_t> tmp(tensor->numel());
        read_bytes(tmp.data());
        for (size_t i = 0; i < tmp.size(); ++i) {
            out[i] = static_cast<int64_t>(tmp[i]);
        }
    } else {
        throw std::invalid_argument("Gemma4Model: expected int32 or int64 metadata tensor");
    }
    return out;
}

} // namespace

Gemma4TextModel::Gemma4TextModel(infinilm::runtime::Runtime &runtime,
                                 std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const engine::distributed::RankInfo &rank_info,
                                 backends::AttentionBackend attention_backend,
                                 const layers::moe::EPConfig &ep_config) {
    (void)ep_config;
    models::validate_runtime_construction(model_config, runtime, rank_info);
    spec_ = parse_gemma4_text_spec(model_config);

    const auto dtype = model_config->get_dtype();
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
    const auto [begin, end] = parallel::Topology(
                                  static_cast<size_t>(rank_info.tp_size), rank_info.pp_size, rank_info.pp_stage)
                                  .layer_range(spec_.num_hidden_layers);
    local_layer_begin_ = begin;
    local_layer_end_ = end;

    try {
        if (pp_stage_ == 0) {
            embed_tokens_ = this->bind_child(
                "embed_tokens",
                std::make_shared<infinilm::layers::Embedding>(
                    runtime, spec_.vocab_size, spec_.hidden_size, std::nullopt, dtype));
            embed_scale_tensor_ = ops::make_filled_param(
                runtime, {1}, dtype, std::sqrt(static_cast<double>(spec_.hidden_size)));

            if (spec_.hidden_size_per_layer_input > 0) {
                const size_t total_ple = spec_.num_hidden_layers * spec_.hidden_size_per_layer_input;
                embed_tokens_per_layer_ = this->bind_child(
                    "embed_tokens_per_layer",
                    std::make_shared<infinilm::layers::Embedding>(
                        runtime, spec_.vocab_size_per_layer_input, total_ple, std::nullopt, dtype));
                ple_embed_scale_tensor_ = ops::make_filled_param(
                    runtime, {1}, dtype, std::sqrt(static_cast<double>(spec_.hidden_size_per_layer_input)));
                per_layer_model_projection_ = this->bind_child(
                    "per_layer_model_projection",
                    std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                        runtime, spec_.hidden_size, total_ple, false /* bias */, dtype));
                ple_proj_scale_tensor_ = ops::make_filled_param(
                    runtime, {1}, dtype, 1.0 / std::sqrt(static_cast<double>(spec_.hidden_size)));
                ple_combine_scale_tensor_ = ops::make_filled_param(
                    runtime, {1}, dtype, 1.0 / std::sqrt(2.0));
                per_layer_projection_norm_ = this->bind_child(
                    "per_layer_projection_norm",
                    std::make_shared<infinilm::layers::RMSNorm>(
                        runtime, spec_.hidden_size_per_layer_input, spec_.rms_norm_eps, dtype));
            }
        }

        // Shared dual RoPE caches: one for sliding_attention, one for full_attention.
        // Cap initial table at 32768 if max_position_embeddings is huge to keep startup fast,
        // while still covering long-context prompts cleanly.
        const size_t rope_max_len = std::min<size_t>(spec_.max_position_embeddings, 65536);
        sliding_rope_ = std::make_shared<Gemma4RoPE>(
            runtime, spec_.head_dim, rope_max_len,
            spec_.sliding_rope_theta, spec_.sliding_partial_rotary_factor, spec_.sliding_is_proportional, dtype);
        full_rope_ = std::make_shared<Gemma4RoPE>(
            runtime, spec_.global_head_dim, rope_max_len,
            spec_.full_rope_theta, spec_.full_partial_rotary_factor, spec_.full_is_proportional, dtype);

        layers_.reserve(local_layer_end_ - local_layer_begin_);
        for (size_t i = local_layer_begin_; i < local_layer_end_; ++i) {
            const auto &ls = spec_.layers[i];
            auto rope = ls.is_sliding ? sliding_rope_ : full_rope_;
            layers_.push_back(this->bind_child(
                "layers." + std::to_string(i),
                std::make_shared<Gemma4TextDecoderLayer>(
                    runtime, model_config, spec_, ls, std::move(rope), rank_info, attention_backend)));
        }

        if (pp_stage_ + 1 == pp_size_) {
            norm_ = this->bind_child(
                "norm",
                std::make_shared<infinilm::layers::RMSNorm>(runtime, spec_.hidden_size, spec_.rms_norm_eps, dtype));
        }

        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            runtime.syncStream();
        }
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

infinilm::runtime::Tensor Gemma4TextModel::embed_tokens(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input_ids) const {
    auto emb = embed_tokens_->forward(runtime, input_ids);
    return ops::mul(runtime, emb, embed_scale_tensor_);
}

infinilm::runtime::Tensor Gemma4TextModel::get_per_layer_inputs(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &input_ids) const {
    auto tok = embed_tokens_per_layer_->forward(runtime, input_ids);
    tok = ops::mul(runtime, tok, ple_embed_scale_tensor_);
    return tok->view({input_ids->size(0), input_ids->size(1), spec_.num_hidden_layers, spec_.hidden_size_per_layer_input});
}

infinilm::runtime::Tensor Gemma4TextModel::project_per_layer_inputs(
    infinilm::runtime::Runtime &runtime,
    const infinilm::runtime::Tensor &inputs_embeds,
    const infinilm::runtime::Tensor &per_layer_inputs) const {
    auto embeds_mut = inputs_embeds;
    auto proj = per_layer_model_projection_->forward(runtime, embeds_mut);
    ops::mul_inplace(runtime, proj, ple_proj_scale_tensor_);
    const size_t bs = inputs_embeds->size(0);
    const size_t seq = inputs_embeds->size(1);
    const size_t tokens = bs * seq;
    // Keep 3D [tokens, num_layers, ple_dim] for RmsNorm/Add/Mul compatibility across all backends.
    proj = proj->view({tokens, spec_.num_hidden_layers, spec_.hidden_size_per_layer_input});
    proj = per_layer_projection_norm_->forward(runtime, proj);
    if (!per_layer_inputs) {
        return proj->view({bs, seq, spec_.num_hidden_layers, spec_.hidden_size_per_layer_input});
    }
    auto ple_3d = per_layer_inputs->view({tokens, spec_.num_hidden_layers, spec_.hidden_size_per_layer_input});
    auto combined = ops::add(runtime, proj, ple_3d);
    ops::mul_inplace(runtime, combined, ple_combine_scale_tensor_);
    return combined->view({bs, seq, spec_.num_hidden_layers, spec_.hidden_size_per_layer_input});
}

infinilm::runtime::Tensor Gemma4TextModel::forward(
    const InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    const auto &input_ids = input.input_ids.value();
    const auto &positions = input.position_ids.value();

    auto inputs_embeds = embed_tokens(runtime, input_ids);
    infinilm::runtime::Tensor per_layer_inputs;
    if (spec_.hidden_size_per_layer_input > 0) {
        per_layer_inputs = get_per_layer_inputs(runtime, input_ids);
        per_layer_inputs = project_per_layer_inputs(runtime, inputs_embeds, per_layer_inputs);
    }
    return forward_embeds(inputs_embeds, per_layer_inputs, positions, forward_context);
}

infinilm::runtime::Tensor Gemma4TextModel::forward_embeds(
    const infinilm::runtime::Tensor &inputs_embeds,
    const infinilm::runtime::Tensor &per_layer_inputs,
    const infinilm::runtime::Tensor &positions,
    global_state::ForwardContext &forward_context,
    const Gemma4VisionSpans *vision_spans) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states = inputs_embeds;

    for (size_t idx = 0; idx < layers_.size(); ++idx) {
        const size_t global_layer_idx = local_layer_begin_ + idx;
        infinilm::runtime::Tensor layer_ple;
        if (per_layer_inputs) {
            layer_ple = per_layer_inputs->narrow({{2, global_layer_idx, 1}})
                            ->squeeze(2)
                            ->contiguous(runtime);
        }
        hidden_states = layers_[idx]->forward(positions, hidden_states, layer_ple, forward_context, vision_spans);
    }

    if (norm_) {
        hidden_states = norm_->forward(runtime, hidden_states);
    }
    return hidden_states;
}

Gemma4Model::Gemma4Model(infinilm::runtime::Runtime &runtime,
                         std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const engine::distributed::RankInfo &rank_info,
                         backends::AttentionBackend attention_backend,
                         const layers::moe::EPConfig &ep_config)
    : model_config_(model_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    language_model_ = this->bind_child(
        "language_model",
        std::make_shared<Gemma4TextModel>(runtime, model_config, rank_info, attention_backend, ep_config));

    const auto dtype = model_config->get_dtype();
    const auto &cfg = model_config->get_config_json();
    const size_t text_hidden = language_model_->spec().hidden_size;

    if (cfg.contains("vision_config") && cfg.at("vision_config").is_object()) {
        const auto &vcfg = cfg.at("vision_config");
        vision_tower_ = this->bind_child(
            "vision_tower", std::make_shared<Gemma4VisionModel>(runtime, vcfg, dtype));
        const size_t v_hidden = vcfg.value("hidden_size", size_t{768});
        const double v_eps = vcfg.value("rms_norm_eps", 1e-6);
        embed_vision_ = this->bind_child(
            "embed_vision", std::make_shared<Gemma4MultimodalEmbedder>(runtime, v_hidden, text_hidden, v_eps, dtype));
    }

    if (cfg.contains("audio_config") && cfg.at("audio_config").is_object()) {
        const auto &acfg = cfg.at("audio_config");
        audio_tower_ = this->bind_child(
            "audio_tower", std::make_shared<Gemma4AudioModel>(runtime, acfg, dtype));
        const size_t a_out = acfg.value("output_proj_dims", acfg.value("hidden_size", size_t{1536}));
        const double a_eps = acfg.value("rms_norm_eps", 1e-6);
        embed_audio_ = this->bind_child(
            "embed_audio", std::make_shared<Gemma4MultimodalEmbedder>(runtime, a_out, text_hidden, a_eps, dtype));
    }
}

infinilm::runtime::Tensor Gemma4Model::forward(
    const InfinilmModel::Input &input,
    global_state::ForwardContext &forward_context) const {
    if (!input.pixel_values.has_value() || input.pixel_values->empty()) {
        return language_model_->forward(input, forward_context);
    }

    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    const auto &input_ids = input.input_ids.value();
    const auto &positions = input.position_ids.value();
    const auto &mm_payloads = input.pixel_values.value();
    const auto &mm_grids = input.image_grid_thw.value();
    const auto &mm_bounds = input.image_bound.value();

    std::vector<int64_t> offsets;
    if (input.input_offsets.has_value()) {
        offsets = read_int_vector_host(runtime, input.input_offsets.value());
    } else {
        offsets = {0, static_cast<int64_t>(input_ids->size(1))};
    }

    // Replace multimodal placeholder token IDs with pad_token_id before embedding & PLE lookup
    // (exact match to HF Gemma4Model.forward lines 2378-2389).
    auto ids_host = read_int_vector_host(runtime, input_ids);
    const int64_t pad_id = language_model_->spec().pad_token_id;
    const auto &req_ids_opt = forward_context.mm_metadata.image_req_ids;

    struct SpanRecord {
        size_t global_start;
        size_t span_len;
        int64_t modality; // >= 1: image (t, grid_h, grid_w), 0: audio (0, num_frames, feat_dim)
        size_t d1;
        size_t d2;
    };
    std::vector<SpanRecord> spans;
    spans.reserve(mm_payloads.size());

    for (size_t idx = 0; idx < mm_payloads.size(); ++idx) {
        const size_t req_id = (req_ids_opt.has_value() && idx < req_ids_opt->size())
                                ? req_ids_opt->at(idx)
                                : 0;
        const size_t req_start = static_cast<size_t>(offsets.at(req_id));
        auto bound = read_int_vector_host(runtime, mm_bounds.at(idx));
        auto grid = read_int_vector_host(runtime, mm_grids.at(idx));
        const size_t local_start = static_cast<size_t>(bound.at(0));
        const size_t local_end = static_cast<size_t>(bound.at(1));
        const size_t g_start = req_start + local_start;
        const size_t span_len = local_end - local_start;
        for (size_t pos = g_start; pos < g_start + span_len && pos < ids_host.size(); ++pos) {
            ids_host[pos] = pad_id;
        }
        spans.push_back({g_start, span_len, grid.at(0), static_cast<size_t>(grid.at(1)), static_cast<size_t>(grid.at(2))});
    }

    // Upload masked llm_input_ids matching input_ids dtype
    const auto cpu = infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu};
    auto llm_input_ids = infinilm::runtime::Tensor::empty(runtime, input_ids->shape(), input_ids->dtype());
    if (input_ids->dtype() == infinilm::runtime::DataType::kInt64) {
        llm_input_ids->copy_from(runtime, infinilm::runtime::Tensor::from_blob(ids_host.data(), input_ids->shape(), input_ids->dtype(), cpu));
    } else {
        std::vector<int32_t> tmp(ids_host.size());
        for (size_t i = 0; i < ids_host.size(); ++i) {
            tmp[i] = static_cast<int32_t>(ids_host[i]);
        }
        llm_input_ids->copy_from(runtime, infinilm::runtime::Tensor::from_blob(tmp.data(), input_ids->shape(), input_ids->dtype(), cpu));
    }

    auto inputs_embeds = language_model_->embed_tokens(runtime, llm_input_ids);
    infinilm::runtime::Tensor token_ple;
    if (language_model_->spec().hidden_size_per_layer_input > 0) {
        token_ple = language_model_->get_per_layer_inputs(runtime, llm_input_ids);
    }

    // Run Vision / Audio encoders and scatter soft tokens into inputs_embeds
    Gemma4VisionSpans vision_spans;
    for (size_t idx = 0; idx < mm_payloads.size(); ++idx) {
        const auto &sp = spans[idx];
        infinilm::runtime::Tensor mm_embeds;
        if (sp.modality >= 1) {
            if (!vision_tower_ || !embed_vision_) {
                throw std::runtime_error("Gemma4Model: vision_tower is not initialized");
            }
            auto vis_feat = vision_tower_->forward(runtime, mm_payloads.at(idx), sp.d1, sp.d2);
            mm_embeds = embed_vision_->forward(runtime, vis_feat);
        } else {
            if (!audio_tower_ || !embed_audio_) {
                throw std::runtime_error("Gemma4Model: audio_tower is not initialized");
            }
            auto aud_feat = audio_tower_->forward(runtime, mm_payloads.at(idx));
            mm_embeds = embed_audio_->forward(runtime, aud_feat);
        }
        const size_t copy_len = std::min<size_t>(sp.span_len, mm_embeds->size(0));
        if (sp.modality >= 1 && copy_len > 0) {
            vision_spans.emplace_back(sp.global_start, copy_len);
        }
        inputs_embeds->narrow({{1, sp.global_start, copy_len}})
            ->copy_from(runtime, mm_embeds->narrow({{0, 0, copy_len}})->unsqueeze(0));
    }

    infinilm::runtime::Tensor per_layer_inputs;
    if (language_model_->spec().hidden_size_per_layer_input > 0) {
        per_layer_inputs = language_model_->project_per_layer_inputs(runtime, inputs_embeds, token_ple);
    }

    return language_model_->forward_embeds(
        inputs_embeds, per_layer_inputs, positions, forward_context,
        vision_spans.empty() ? nullptr : &vision_spans);
}

template <typename ModelT>
Gemma4CausalLMBase<ModelT>::Gemma4CausalLMBase(
    infinilm::runtime::Runtime &runtime,
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const engine::distributed::RankInfo &rank_info,
    backends::AttentionBackend attention_backend,
    const layers::moe::EPConfig &ep_config) {
    models::validate_runtime_construction(model_config, runtime, rank_info);
    model_config_ = model_config;
    const auto spec = parse_gemma4_text_spec(model_config);
    final_logit_softcapping_ = spec.final_logit_softcapping;
    pp_size_ = static_cast<size_t>(rank_info.pp_size);
    pp_stage_ = static_cast<size_t>(rank_info.pp_stage);
    const auto dtype = model_config->get_dtype();

    int32_t shift_value = -1;
    infinilm::runtime::Tensor shift_host;
    try {
        model_ = this->bind_child(
            "model",
            std::make_shared<ModelT>(runtime, model_config, rank_info, attention_backend, ep_config));
        if (pp_stage_ + 1 == pp_size_) {
            lm_head_ = this->bind_child(
                "lm_head",
                std::make_shared<infinilm::layers::linear::ReplicatedLinear>(
                    runtime, spec.hidden_size, spec.vocab_size, false /* bias */, dtype));
            shift_host = infinilm::runtime::Tensor::from_blob(
                &shift_value, {1}, infinilm::runtime::DataType::kInt32,
                infinilm::runtime::Device{infinilm::runtime::Device::Type::kCpu});
            last_token_shift_ = infinilm::runtime::Tensor::empty(runtime, {1}, infinilm::runtime::DataType::kInt32);
            last_token_shift_->copy_from(runtime, shift_host);
            if (final_logit_softcapping_ > 0.0) {
                softcap_inv_half_ = ops::make_filled_param(runtime, {1}, dtype, 2.0 / final_logit_softcapping_);
                softcap_two_ = ops::make_filled_param(runtime, {1}, dtype, 2.0 * final_logit_softcapping_);
                softcap_neg_ = ops::make_filled_param(runtime, {1}, dtype, -final_logit_softcapping_);
            }
        }
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            runtime.syncStream();
        }
    } catch (...) {
        if (runtime.device().type() != infinilm::runtime::Device::Type::kCpu) {
            try {
                runtime.syncStream();
            } catch (...) {
                std::terminate();
            }
        }
        throw;
    }
}

template <typename ModelT>
InfinilmModel::Output Gemma4CausalLMBase<ModelT>::forward(
    const Input &input,
    global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states = model_->forward(input, forward_context);
    if (pp_stage_ + 1 != pp_size_) {
        return {infinilm::runtime::Tensor(), hidden_states};
    }

    auto lm_head_input = hidden_states;
    if (!input.sample_all_positions && input.input_offsets.has_value()) {
        const size_t num_requests = input.input_offsets.value()->numel() - 1;
        const bool is_packed_prefill = hidden_states->ndim() == 3
                                    && hidden_states->size(0) == 1
                                    && hidden_states->size(1) > num_requests;
        if (is_packed_prefill) {
            auto end_offsets = input.input_offsets.value()->narrow({{0, 1, num_requests}});
            auto last_pos = infinilm::runtime::Tensor::empty(runtime, end_offsets->shape(), end_offsets->dtype());
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()),
                end_offsets, last_token_shift_, last_pos);
            auto packed_hidden = hidden_states->view({hidden_states->size(1), hidden_states->size(2)});
            auto gathered = infinilm::runtime::Tensor::empty(
                runtime, {num_requests, hidden_states->size(2)}, packed_hidden->dtype());
            infinilm::execution::submit<infini::ops::Embedding>(
                runtime, infini::ops::Embedding::DefaultConfig(runtime.device().type()),
                last_pos, packed_hidden, gathered);
            lm_head_input = gathered->view({1, num_requests, hidden_states->size(2)});
        }
    }

    auto logits = lm_head_->forward(runtime, lm_head_input);
    if (final_logit_softcapping_ > 0.0) {
        ops::mul_inplace(runtime, logits, softcap_inv_half_);
        infinilm::execution::submit<infini::ops::Sigmoid>(
            runtime, infini::ops::Sigmoid::DefaultConfig(runtime.device().type()), logits, logits);
        ops::mul_inplace(runtime, logits, softcap_two_);
        ops::add_inplace(runtime, logits, softcap_neg_);
    }
    return {logits, hidden_states};
}

template <typename ModelT>
void Gemma4CausalLMBase<ModelT>::reset_cache(
    const cache::CacheConfig *cache_config,
    global_state::ForwardContext &forward_context,
    backends::AttentionBackend attention_backend,
    const engine::distributed::RankInfo &rank_info) {
    cache::validate_backend_cache_config(cache_config, attention_backend);
    if (cache_config == nullptr) {
        cache_config_.reset();
        forward_context.kv_cache_vec.clear();
        return;
    }
    auto &runtime = forward_context.require_runtime();
    cache::validate_cache_runtime(runtime, rank_info);
    cache_config_ = cache_config->unique_copy();
    forward_context.kv_cache_vec.clear();
    forward_context.kv_cache_vec = gemma4_allocate_kv_cache_tensors(
        runtime, rank_info, cache_config, model_config_, attention_backend);
}

template class Gemma4CausalLMBase<Gemma4Model>;
template class Gemma4CausalLMBase<Gemma4TextModel>;

} // namespace infinilm::models::gemma4

namespace {
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    gemma4,
    infinilm::models::gemma4::Gemma4ForConditionalGeneration,
    infinilm::models::gemma4::create_gemma4_model_config);

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    gemma4_text,
    infinilm::models::gemma4::Gemma4ForCausalLM,
    infinilm::models::gemma4::create_gemma4_text_model_config);
} // namespace
