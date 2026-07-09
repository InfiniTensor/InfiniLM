#include "deepseek_v2_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "infinicore/context/context.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v2 {

DeepseekV2Model::DeepseekV2Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<DeepseekV2DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, rms_norm_eps, dtype, device);
}

infinicore::Tensor DeepseekV2Model::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto positions = input.position_ids.value();
    auto hidden_states = embed_tokens_->forward(input_ids);

    infinicore::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(positions, hidden_states, residual);
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

DeepseekV2ForCausalLM::DeepseekV2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

infinilm::InfinilmModel::Output DeepseekV2ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);

    const auto &hidden_shape = hidden_states->shape();
    if (hidden_shape.size() >= 2 && hidden_shape[0] * hidden_shape[1] > 4096) {
        // Release cached prefill temporaries so lm_head can allocate full logits for large batches.
        infinicore::context::trimMemory();
    }

    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void DeepseekV2ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
    if (!infinilm_config.use_mla || cache_config == nullptr) {
        InfinilmModel::reset_cache(cache_config);
        return;
    }

    cache_config_ = cache_config->unique_copy();

    auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    kv_cache_vec.clear();
    auto new_kv_cache_vec = deepseek_v2_allocate_kv_cache_tensors(cache_config, model_config_, infinilm_config.attention_backend);
    kv_cache_vec = std::move(new_kv_cache_vec);
}

std::shared_ptr<infinilm::config::ModelConfig> create_deepseek_v2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string model_type = model_config->get<std::string>("model_type");
    if ("deepseek_v2" != model_type) {
        throw std::runtime_error("create_deepseek_v2_model_config: model_type is not deepseek_v2");
    }

    auto &config_json = model_config->get_config_json();
    const size_t q_head_dim = config_json.at("qk_nope_head_dim").get<size_t>() + config_json.at("qk_rope_head_dim").get<size_t>();
    config_json["head_dim"] = q_head_dim;

    const size_t qk_rope_head_dim = config_json.at("qk_rope_head_dim").get<size_t>();
    config_json["partial_rotary_factor"] = static_cast<double>(qk_rope_head_dim) / static_cast<double>(q_head_dim);

    config_json["num_experts"] = config_json.value("n_routed_experts", 0);
    config_json["mlp_bias"] = false;
    if (!config_json.contains("attention_output_bias")) {
        config_json["attention_output_bias"] = config_json.value("attention_bias", false);
    }
    if (!config_json.contains("dtype") && config_json.contains("torch_dtype")) {
        config_json["dtype"] = config_json["torch_dtype"];
    }

    // Use GPT-J style for DeepseekV2
    model_config->set_rope_algo(infinicore::nn::RoPE::Algo::GPT_J);

    return model_config;
}

} // namespace infinilm::models::deepseek_v2

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    deepseek_v2,
    infinilm::models::deepseek_v2::DeepseekV2ForCausalLM,
    infinilm::models::deepseek_v2::create_deepseek_v2_model_config);
} // namespace
