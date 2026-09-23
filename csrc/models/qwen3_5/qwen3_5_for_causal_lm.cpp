#include "qwen3_5_for_causal_lm.hpp"

#include "../../cache/kv_cache.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include <infinicore/ops/select_last_token_hidden.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinilm::models::qwen3_5 {

Qwen35ForCausalLM::Qwen35ForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device) {
    model_config_ = model_config;
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const auto &dtype = model_config->get_dtype();

    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    const auto &rank = global_state::get_tensor_model_parallel_rank_info();
    INFINICORE_NN_MODULE_INIT(
        lm_head, hidden_size, vocab_size, dtype, device, rank.tp_rank, rank.tp_size, rank.comm);
    if (model_config->get_or<bool>("enable_mtp", false)) {
        INFINICORE_NN_MODULE_INIT(mtp, model_config, device);
    }
}

InfinilmModel::Output Qwen35ForCausalLM::forward(
    const InfinilmModel::Input &input) const {
    infinicore::Tensor hidden_states;
    if (input.target_hidden_states.has_value()) {
        if (!mtp_) {
            throw std::runtime_error("Qwen MTP weights must be enabled before draft execution.");
        }
        hidden_states = mtp_->forward(model_->embed_input_ids(input.input_ids.value()),
                                      input.target_hidden_states.value(), input.position_ids.value());
    } else {
        hidden_states = model_->forward(input);
    }
    auto head_input = hidden_states;
    const bool packed_greedy = input.greedy_output && input.block_tables && input.input_offsets
                            && hidden_states->size(0) == 1
                            && hidden_states->device().getType() == infinicore::Device::Type::NVIDIA;
    if (packed_greedy && !input.sample_all_positions
        && hidden_states->size(1) != input.input_offsets.value()->numel() - 1) {
        head_input = infinicore::Tensor::empty(
            {1, input.input_offsets.value()->numel() - 1, hidden_states->size(2)},
            hidden_states->dtype(), hidden_states->device());
        infinicore::op::select_last_token_hidden_(head_input, hidden_states, input.input_offsets.value());
    }
    if (packed_greedy) {
        return {{}, mtp_ ? hidden_states : infinicore::Tensor{}, lm_head_->top_tokens(head_input)};
    }
    auto logits = lm_head_->forward(head_input);
    if (mtp_) {
        return {logits, hidden_states};
    }
    return {logits};
}

void Qwen35ForCausalLM::reset_cache(
    const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
    } else {
        cache_config_ = cache_config->unique_copy();
    }
    model_->reset_cache(cache_config);
    if (mtp_ && cache_config != nullptr) {
        const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config);
        if (paged == nullptr) {
            throw std::runtime_error("Qwen MTP requires paged KV storage.");
        }
        const auto head_dim = model_config_->get<size_t>("head_dim");
        const auto heads = model_config_->get<size_t>("num_key_value_heads");
        auto &context = global_state::get_forward_context();
        context.kv_cache_vec.push_back(cache::PagedKVCache::create_layer_kv_cache(
            head_dim, head_dim, heads, heads, model_config_->get_kv_cache_dtype(), *paged));
        context.conv_state_vec.emplace_back();
        context.ssm_state_vec.emplace_back();
    }
}

std::shared_ptr<infinilm::config::ModelConfig> prepare_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    nlohmann::json &config_json = model_config->get_config_json();
    if (config_json.contains("text_config") && config_json["text_config"].is_object()) {
        const nlohmann::json &text_config_json = config_json["text_config"];
        for (auto it = text_config_json.begin(); it != text_config_json.end(); ++it) {
            if (!config_json.contains(it.key())) {
                config_json[it.key()] = it.value();
            }
        }
        if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
            config_json["dtype"] = config_json["torch_dtype"];
        }
    }
    if (!config_json.contains("position_id_axes")) {
        size_t position_id_axes = 1;
        if (config_json.contains("rope_parameters")
            && config_json["rope_parameters"].is_object()) {
            const auto &rope_parameters = config_json["rope_parameters"];
            if (rope_parameters.contains("mrope_section")
                && rope_parameters["mrope_section"].is_array()
                && !rope_parameters["mrope_section"].empty()) {
                position_id_axes = rope_parameters["mrope_section"].size();
            }
        }
        config_json["position_id_axes"] = position_id_axes;
    }
    if (!config_json.contains("rope_theta") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("rope_theta")) {
        // Normalize the nested HuggingFace field for the Qwen3.5 attention module.
        config_json["rope_theta"] = config_json["rope_parameters"]["rope_theta"];
    }
    if (!config_json.contains("partial_rotary_factor") && config_json.contains("rope_parameters") && config_json["rope_parameters"].is_object() && config_json["rope_parameters"].contains("partial_rotary_factor")) {
        config_json["partial_rotary_factor"] = config_json["rope_parameters"]["partial_rotary_factor"];
    }
    if (!config_json.contains("layer_types")) {
        const size_t full_attention_interval = model_config->get<size_t>("full_attention_interval");
        if (full_attention_interval == 0) {
            throw std::runtime_error("Qwen3.5 full_attention_interval must be positive");
        }
        const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        std::vector<std::string> layer_types;
        layer_types.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layer_types.push_back(
                (i + 1) % full_attention_interval == 0
                    ? "full_attention"
                    : "linear_attention");
        }
        config_json["layer_types"] = std::move(layer_types);
    }
    if (!config_json.contains("attention_bias")) {
        config_json["attention_bias"] = false;
    }
    return model_config;
}

std::shared_ptr<infinilm::config::ModelConfig> create_qwen3_5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("qwen3_5" != model_type) {
        throw std::runtime_error("infinilm::models::qwen3_5::create_qwen3_5_model_config: model_type is not qwen3_5");
    }
    return prepare_qwen3_5_model_config(model_config);
}

} // namespace infinilm::models::qwen3_5

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    qwen3_5,
    infinilm::models::qwen3_5::Qwen35ForCausalLM,
    infinilm::models::qwen3_5::create_qwen3_5_model_config);
} // namespace
