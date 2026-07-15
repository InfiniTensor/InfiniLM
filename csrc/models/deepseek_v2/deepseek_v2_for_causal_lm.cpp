#include "deepseek_v2_for_causal_lm.hpp"

#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::deepseek_v2 {

DeepseekV2Model::DeepseekV2Model(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device) {
    const auto &dtype{model_config->get_dtype()};
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    INFINICORE_NN_MODULE_INIT(
        embed_tokens, model_config->get<size_t>("vocab_size"), hidden_size, std::nullopt, dtype, device);
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    layers_.reserve(num_layers);
    for (size_t layer = 0; layer < num_layers; ++layer) {
        layers_.push_back(this->register_module<DeepseekV2DecoderLayer>(
            "layers." + std::to_string(layer), model_config, layer, device));
    }
    INFINICORE_NN_MODULE_INIT(
        norm, hidden_size, model_config->get<double>("rms_norm_eps"), dtype, device);
}

infinicore::Tensor DeepseekV2Model::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = embed_tokens_->forward(input.input_ids.value());
    infinicore::Tensor residual;
    for (const auto &layer : layers_) {
        layer->forward(input.position_ids.value(), hidden_states, residual);
    }
    norm_->forward_inplace(hidden_states, residual);
    return hidden_states;
}

DeepseekV2ForCausalLM::DeepseekV2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                             const infinicore::Device &device) {
    model_config_ = model_config;
    INFINICORE_NN_MODULE_INIT(model, model_config, device);
    INFINICORE_NN_MODULE_INIT(lm_head,
                              model_config->get<size_t>("hidden_size"),
                              model_config->get<size_t>("vocab_size"),
                              false,
                              model_config->get_dtype(),
                              device);
}

infinilm::InfinilmModel::Output DeepseekV2ForCausalLM::forward(const infinilm::InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    return {lm_head_->forward(hidden_states)};
}

void DeepseekV2ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        InfinilmModel::reset_cache(cache_config);
        return;
    }
    cache_config_ = cache_config->unique_copy();
    global_state::get_forward_context().kv_cache_vec = deepseek_v2_allocate_kv_cache_tensors(
        cache_config, model_config_, global_state::get_infinilm_config().attention_backend);
}

std::shared_ptr<infinilm::config::ModelConfig>
create_deepseek_v2_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    if (model_config->get<std::string>("model_type") != "deepseek_v2") {
        throw std::runtime_error("create_deepseek_v2_model_config: model_type is not deepseek_v2");
    }
    auto &json = model_config->get_config_json();
    const size_t nope_dim = json.at("qk_nope_head_dim").get<size_t>();
    const size_t rope_dim = json.at("qk_rope_head_dim").get<size_t>();
    json["head_dim"] = nope_dim + rope_dim;
    json["partial_rotary_factor"] = static_cast<double>(rope_dim) / static_cast<double>(nope_dim + rope_dim);
    json["num_experts"] = json.value("n_routed_experts", 0);
    json["mlp_bias"] = false;
    json["moe_router_backend"] = "vllm_topk";
    if (!json.contains("attention_output_bias")) {
        json["attention_output_bias"] = json.value("attention_bias", false);
    }
    if (!json.contains("dtype") && json.contains("torch_dtype")) {
        json["dtype"] = json["torch_dtype"];
    }
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
