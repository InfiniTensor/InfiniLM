#include "minimax_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace infinilm::models::minimax {

MiniMaxModel::MiniMaxModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get_or<double>("rms_norm_eps", 1e-5);

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<MiniMaxDecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor MiniMaxModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    if (input_ids->shape().size() == 1) {
        input_ids = input_ids->view({1, input_ids->shape()[0]});
    }
    auto hidden_states = embed_tokens_->forward(input_ids);
    auto positions = input.position_ids.value();
    for (const auto &layer : layers_) {
        hidden_states = layer->forward(positions, hidden_states);
    }
    return norm_->forward(hidden_states);
}

MiniMaxForCausalLM::MiniMaxForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                       const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output MiniMaxForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MiniMaxForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();

    auto &forward_context = infinilm::global_state::get_forward_context();
    forward_context.kv_cache_vec.clear();
    forward_context.conv_state_vec.clear();
    forward_context.ssm_state_vec.clear();

    const size_t num_hidden_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t hidden_size = model_config_->get<size_t>("hidden_size");
    const size_t total_num_heads = model_config_->get<size_t>("num_attention_heads");
    const size_t head_dim = model_config_->get_or<size_t>("head_dim", 0);
    const size_t resolved_head_dim = head_dim != 0 ? head_dim : hidden_size / total_num_heads;
    const size_t num_kv_heads = model_config_->get_or<size_t>("num_key_value_heads", total_num_heads);
    const size_t max_position_embeddings = model_config_->get_or<size_t>("max_position_embeddings", 8192);
    const auto &dtype{model_config_->get_dtype()};
    const auto &kv_cache_dtype{model_config_->get_kv_cache_dtype()};
    const std::vector<std::string> layer_types = model_config_->get<std::vector<std::string>>("layer_types");
    const backends::AttentionBackend attention_backend = infinilm::global_state::get_infinilm_config().attention_backend;

    forward_context.kv_cache_vec.reserve(num_hidden_layers);
    forward_context.ssm_state_vec.reserve(num_hidden_layers);

    auto allocate_linear_state = [&](size_t layer_idx, size_t pool_size) {
        auto state = cache::MambaCache::create_layer_ssm_state(
            resolved_head_dim,
            resolved_head_dim,
            total_num_heads,
            total_num_heads,
            infinicore::DataType::F32,
            pool_size);
        forward_context.kv_cache_vec.emplace_back();
        forward_context.ssm_state_vec.push_back(std::move(state));
    };

    auto allocate_static_full_attention = [&](size_t layer_idx, const cache::StaticKVCacheConfig &config) {
        auto kv_cache = cache::StaticKVCache::create_layer_kv_cache(
            resolved_head_dim,
            resolved_head_dim,
            num_kv_heads,
            num_kv_heads,
            max_position_embeddings,
            kv_cache_dtype,
            config);
        forward_context.kv_cache_vec.push_back(std::move(kv_cache));
        forward_context.ssm_state_vec.emplace_back();
    };

    auto allocate_paged_full_attention = [&](size_t layer_idx, const cache::PagedKVCacheConfig &config) {
        auto kv_cache = cache::PagedKVCache::create_layer_kv_cache(
            resolved_head_dim,
            resolved_head_dim,
            num_kv_heads,
            num_kv_heads,
            kv_cache_dtype,
            config);
        forward_context.kv_cache_vec.push_back(std::move(kv_cache));
        forward_context.ssm_state_vec.emplace_back();
    };

    switch (attention_backend) {
    case backends::AttentionBackend::STATIC_ATTN: {
        auto static_kv_cache_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config);
        if (nullptr == static_kv_cache_config) {
            throw std::runtime_error("infinilm::models::minimax::MiniMaxForCausalLM: invalid static kv cache config type");
        }
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            if ("linear_attention" == layer_types[layer_idx]) {
                allocate_linear_state(layer_idx, static_kv_cache_config->max_batch_size());
            } else {
                allocate_static_full_attention(layer_idx, *static_kv_cache_config);
            }
        }
        break;
    }
    case backends::AttentionBackend::FLASH_ATTN:
    case backends::AttentionBackend::PAGED_ATTN: {
        auto paged_kv_cache_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (nullptr == paged_kv_cache_config) {
            throw std::runtime_error("infinilm::models::minimax::MiniMaxForCausalLM: invalid paged kv cache config type");
        }
        const size_t lightning_pool_size = std::max<size_t>(2, paged_kv_cache_config->num_blocks() / 4);
        for (size_t layer_idx = 0; layer_idx < num_hidden_layers; ++layer_idx) {
            if ("linear_attention" == layer_types[layer_idx]) {
                allocate_linear_state(layer_idx, lightning_pool_size);
            } else {
                allocate_paged_full_attention(layer_idx, *paged_kv_cache_config);
            }
        }
        break;
    }
    default:
        throw std::runtime_error("infinilm::models::minimax::MiniMaxForCausalLM: Unsupported attention backend: " + std::to_string(static_cast<int>(attention_backend)));
    }
}

std::shared_ptr<infinilm::config::ModelConfig> create_minimax_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("minimax" != model_type && "minimax_m2" != model_type) {
        throw std::runtime_error("infinilm::models::minimax::create_minimax_model_config: model_type is not minimax/minimax_m2");
    }

    nlohmann::json &config_json = model_config->get_config_json();

    // --- Normalize basic transformer fields ---
    config_json["num_key_value_heads"] = config_json.value("num_key_value_heads", config_json.value("num_attention_heads", 0));
    if (!config_json.contains("head_dim")) {
        config_json["head_dim"] = config_json["hidden_size"].get<size_t>() / config_json["num_attention_heads"].get<size_t>();
    }
    config_json["rms_norm_eps"] = config_json.value("rms_norm_eps", config_json.value("layer_norm_epsilon", 1e-5));
    config_json["max_position_embeddings"] = config_json.value("max_position_embeddings", config_json.value("max_model_len", 8192));
    // transformers MiniMax keeps rope theta inside `rope_parameters`; flatten it.
    if (!config_json.contains("rope_theta")) {
        if (config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() &&
            config_json["rope_parameters"].contains("rope_theta")) {
            config_json["rope_theta"] = config_json["rope_parameters"]["rope_theta"];
        } else {
            config_json["rope_theta"] = 1000000.0;
        }
    }

    // --- Decode per-layer attention types into `layer_types` ---
    if (!config_json.contains("layer_types")) {
        const size_t num_hidden_layers = config_json["num_hidden_layers"].get<size_t>();
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        if (config_json.contains("attn_type_list") && config_json["attn_type_list"].is_array()) {
            for (const auto &v : config_json["attn_type_list"]) {
                layer_types.push_back(v.get<int>() == 0 ? "linear_attention" : "full_attention");
            }
        } else if (config_json.contains("decoder_attention_types") && config_json["decoder_attention_types"].is_array()) {
            for (const auto &v : config_json["decoder_attention_types"]) {
                layer_types.push_back(v.get<std::string>());
            }
        } else if (config_json.contains("full_attention_interval")) {
            const size_t interval = config_json["full_attention_interval"].get<size_t>();
            for (size_t i = 0; i < num_hidden_layers; ++i) {
                layer_types.push_back(bool((i + 1) % interval) ? "linear_attention" : "full_attention");
            }
        } else {
            // MiniMax-01 default: one softmax attention layer after every 7 lightning layers.
            for (size_t i = 0; i < num_hidden_layers; ++i) {
                layer_types.push_back(bool((i + 1) % 8) ? "linear_attention" : "full_attention");
            }
        }
        if (layer_types.size() != num_hidden_layers) {
            throw std::runtime_error("infinilm::models::minimax::create_minimax_model_config: layer_types length mismatch");
        }
        config_json["layer_types"] = layer_types;
    }

    // --- Lightning attention knobs ---
    config_json["block"] = config_json.value("block", 256);

    // --- MoE defaults ---
    config_json["hidden_act"] = config_json.value("hidden_act", "silu");
    config_json["num_experts_per_tok"] = config_json.value("num_experts_per_tok", 1);
    // HF transformers `minimax` uses `num_local_experts`; normalize the alias.
    if (config_json.contains("num_local_experts") && !config_json.contains("num_experts")) {
        config_json["num_experts"] = config_json["num_local_experts"];
    } else {
        config_json["num_experts"] = config_json.value("num_experts", 1);
    }
    // Expert FFN dim. transformers `minimax` uses `intermediate_size` for the experts.
    config_json["moe_intermediate_size"] = config_json.value("moe_intermediate_size", config_json["intermediate_size"]);
    // transformers renormalizes the top-k softmax weights over the selected experts.
    config_json["norm_topk_prob"] = config_json.value("norm_topk_prob", true);
    config_json["moe_router_backend"] = config_json.value("moe_router_backend", "softmax");
    config_json["shared_intermediate_size"] = config_json.value("shared_intermediate_size", 0);

    return model_config;
}

} // namespace infinilm::models::minimax

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minimax,
    infinilm::models::minimax::MiniMaxForCausalLM,
    infinilm::models::minimax::create_minimax_model_config);
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    minimax_m2,
    infinilm::models::minimax::MiniMaxForCausalLM,
    infinilm::models::minimax::create_minimax_model_config);
} // namespace



