#include "mamba2_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include <algorithm>
#include <cmath>

namespace infinilm::models::mamba2 {
namespace {

class TiedLMHead final : public layers::linear::ReplicatedLinear {
public:
    explicit TiedLMHead(const infinicore::Tensor &weight)
        : layers::linear::ReplicatedLinear(weight->size(1), weight->size(0), false, weight->dtype(), weight->device()) {
        register_parameter("weight", infinicore::nn::Parameter(weight));
    }

    // Preserve the embedding's shared layout when other projections are packed.
    void process_weights_after_loading() override {}
};

} // namespace

std::shared_ptr<config::ModelConfig> create_mamba2_model_config(std::shared_ptr<config::ModelConfig> config) {
    auto &j = config->get_config_json();
    if (j.at("model_type") != "mamba2") {
        throw std::runtime_error("Expected `model_type=mamba2`.");
    }
    for (const auto *key : {"hidden_size", "num_hidden_layers", "vocab_size", "num_heads", "head_dim", "state_size"}) {
        if (config->get<int64_t>(key) <= 0) {
            throw std::runtime_error(std::string("Mamba-2 requires positive `") + key + "`.");
        }
    }
    const auto hidden = config->get<size_t>("hidden_size");
    const auto intermediate = hidden * config->get_or<size_t>("expand", 2);
    const auto heads = config->get<size_t>("num_heads");
    const auto head_dim = config->get<size_t>("head_dim");
    if (config->get_or<int64_t>("expand", 2) <= 0 || heads * head_dim != intermediate
        || config->get<size_t>("state_size") > 256
        || config->get_or<std::string>("hidden_act", "silu") != "silu"
        || config->get_or<size_t>("intermediate_size", intermediate) != intermediate
        || config->get_or<size_t>("n_groups", 1) != 1
        || config->get_or<size_t>("conv_kernel", 4) != 4
        || config->get_or<bool>("use_bias", false)
        || !config->get_or<bool>("use_conv_bias", true)
        || config->get_or<bool>("norm_before_gate", false)
        || !config->get_or<bool>("residual_in_fp32", true)
        || !config->get_or<bool>("rms_norm", true)
        || !config->get_or<bool>("rmsnorm", true)
        || config->get_or<bool>("D_has_hdim", false)
        || config->get_or<size_t>("d_ssm", intermediate) != intermediate
        || config->get_or<size_t>("d_intermediate", 0) != 0
        || (j.contains("attn_layer_idx") && !j["attn_layer_idx"].empty())
        || (j.contains("quantization_config") && !j["quantization_config"].empty())) {
        throw std::runtime_error("Unsupported Mamba-2 configuration; use the checkpoint preparation tool.");
    }
    for (const auto *key : {"dt_limit", "time_step_limit"}) {
        if (j.contains(key)) {
            throw std::runtime_error("Explicit time-step limits are not supported; omit the field for the unbounded Mamba-2 scan.");
        }
    }
    j["intermediate_size"] = intermediate;
    j["layer_norm_epsilon"] = j.value("layer_norm_epsilon", 1e-5);
    j["rms_norm_eps"] = j["layer_norm_epsilon"];
    const double epsilon = config->get<double>("layer_norm_epsilon");
    if (!std::isfinite(epsilon) || epsilon <= 0) {
        throw std::runtime_error("Mamba-2 requires a finite positive normalization epsilon.");
    }
    return config;
}

Mamba2ForCausalLM::Mamba2ForCausalLM(std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device)
    : TextCausalLM(config, device) {
    if (config->get_or<bool>("tie_word_embeddings", true)) {
        lm_head_ = register_module<TiedLMHead>("lm_head", model_->embedding_weight());
    }
}

Mamba2Model::Mamba2Model(std::shared_ptr<config::ModelConfig> config, const infinicore::Device &device)
    : dtype_(config->get_dtype()) {
    const auto &rank = global_state::get_tensor_model_parallel_rank_info();
    if (rank.pp_size != 1) {
        throw std::runtime_error("Mamba-2 currently requires `pp_size=1`.");
    }
    const auto hidden = config->get<size_t>("hidden_size");
    INFINICORE_NN_MODULE_INIT(embeddings, config->get<size_t>("vocab_size"), hidden, std::nullopt, dtype_, device);
    for (size_t i = 0; i < config->get<size_t>("num_hidden_layers"); ++i) {
        layers_.push_back(register_module<Mamba2Block>("layers." + std::to_string(i), config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm_f, hidden, config->get<double>("layer_norm_epsilon"), infinicore::DataType::F32, device);
}

infinicore::Tensor Mamba2Model::forward(const InfinilmModel::Input &input) const {
    if (!input.input_offsets || !input.mamba_init_state_indices || !input.mamba_final_state_indices) {
        throw std::runtime_error("Mamba-2 requires packed offsets and request state indices.");
    }
    auto ids = input.input_ids.value()->view({1, input.input_ids.value()->numel()});
    auto residual = cast_activation(embeddings_->forward(ids), infinicore::DataType::F32);
    for (const auto &layer : layers_) {
        residual = layer->forward(residual);
    }
    return cast_activation(norm_f_->forward(residual), dtype_);
}

void Mamba2ForCausalLM::reset_cache(const cache::CacheConfig *config) {
    auto &context = global_state::get_forward_context();
    context.kv_cache_vec.clear();
    context.conv_state_vec.clear();
    context.ssm_state_vec.clear();
    cache_config_ = config ? config->unique_copy() : nullptr;
    if (config == nullptr) {
        return;
    }
    const auto *paged = dynamic_cast<const cache::PagedKVCacheConfig *>(config);
    if (paged == nullptr) {
        throw std::runtime_error("Mamba-2 requires the paged request-state cache interface.");
    }
    const size_t pool = std::max<size_t>(2, paged->num_blocks() / 4);
    const auto heads = model_config_->get<size_t>("num_heads") / global_state::get_tensor_model_parallel_world_size();
    const auto head_dim = model_config_->get<size_t>("head_dim");
    const auto state_size = model_config_->get<size_t>("state_size");
    const auto conv_dim = heads * head_dim + 2 * state_size;
    const auto device = infinicore::context::getDevice();
    for (size_t i = 0; i < model_config_->get<size_t>("num_hidden_layers"); ++i) {
        context.conv_state_vec.push_back(infinicore::Tensor::zeros({pool, conv_dim, 3}, model_config_->get_dtype(), device));
        context.ssm_state_vec.push_back(infinicore::Tensor::zeros({pool, heads, head_dim, state_size}, infinicore::DataType::F32, device));
    }
}

} // namespace infinilm::models::mamba2

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(mamba2, infinilm::models::mamba2::Mamba2ForCausalLM, infinilm::models::mamba2::create_mamba2_model_config);
} // namespace
