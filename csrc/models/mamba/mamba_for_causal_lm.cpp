#include "mamba_for_causal_lm.hpp"
#include "../models_registry.hpp"
#include <stdexcept>

namespace infinilm::models::mamba {

std::shared_ptr<infinilm::config::ModelConfig>
create_mamba_model_config(std::shared_ptr<infinilm::config::ModelConfig> config) {
    const std::string model_type = config->get<std::string>("model_type");
    if (model_type != "mamba") {
        throw std::runtime_error("mamba config creator called for non-mamba model");
    }
    auto &j = config->get_config_json();
    j["hidden_size"] = j.value("hidden_size", j.value("d_model", 2048));
    j["num_hidden_layers"] = j.value("num_hidden_layers", j.value("n_layer", 48));
    j["intermediate_size"] = j.value("intermediate_size", j["hidden_size"].get<size_t>() * j.value("expand", 2));
    j["layer_norm_epsilon"] = j.value("layer_norm_epsilon", 1e-5);
    j["rms_norm_eps"] = j.value("rms_norm_eps", j["layer_norm_epsilon"]);
    j["state_size"] = j.value("state_size", 16);
    j["conv_kernel"] = j.value("conv_kernel", 4);
    j["time_step_rank"] = j.value("time_step_rank", (j["hidden_size"].get<size_t>() + 15) / 16);
    j["use_bias"] = j.value("use_bias", false);
    j["use_conv_bias"] = j.value("use_conv_bias", true);
    j["max_position_embeddings"] = j.value("max_position_embeddings", 8192);
    return config;
}

MambaMixer::MambaMixer(std::shared_ptr<infinilm::config::ModelConfig> config,
                       size_t layer_idx,
                       const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    intermediate_size_ = config->get<size_t>("intermediate_size");
    state_size_ = config->get<size_t>("state_size");
    time_step_rank_ = config->get<size_t>("time_step_rank");
    conv_kernel_ = config->get<size_t>("conv_kernel");
    const bool use_bias = config->get_or<bool>("use_bias", false);
    const bool use_conv_bias = config->get_or<bool>("use_conv_bias", true);

    INFINICORE_NN_MODULE_INIT(in_proj, hidden_size, intermediate_size_ * 2, use_bias, dtype, device);
    INFINICORE_NN_MODULE_INIT(x_proj, intermediate_size_, time_step_rank_ + state_size_ * 2, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(dt_proj, time_step_rank_, intermediate_size_, true, dtype, device);
    INFINICORE_NN_MODULE_INIT(out_proj, intermediate_size_, hidden_size, use_bias, dtype, device);
    INFINICORE_NN_PARAMETER_INIT(conv1d_weight, ({intermediate_size_, 1, conv_kernel_}, dtype, device));
    if (use_conv_bias) {
        INFINICORE_NN_PARAMETER_INIT(conv1d_bias, ({intermediate_size_}, dtype, device));
    }
    INFINICORE_NN_PARAMETER_INIT(A_log, ({intermediate_size_, state_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(D, ({intermediate_size_}, dtype, device));
    dt_bias_zero_ = infinicore::Tensor::zeros({intermediate_size_}, dtype, device);
}

infinicore::Tensor MambaMixer::forward(const infinicore::Tensor &hidden_states) const {
    auto x_mut = const_cast<infinicore::Tensor &>(hidden_states);
    auto projected = in_proj_->forward(x_mut);
    auto x = projected->narrow({{2, 0, intermediate_size_}})->contiguous();
    auto gate = projected->narrow({{2, intermediate_size_, intermediate_size_}})->contiguous();

    auto &cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    const size_t conv_idx = layer_idx_ * 2;
    const size_t ssm_idx = conv_idx + 1;
    if (cache_vec.size() <= ssm_idx) {
        throw std::runtime_error("MambaMixer: Mamba state cache is not allocated");
    }
    auto conv_state = cache_vec[conv_idx];
    auto ssm_state = cache_vec[ssm_idx];

    auto conv_out = infinicore::Tensor::empty(x->shape(), x->dtype(), x->device());
    infinicore::op::causal_conv1d_(
        conv_out,
        conv_state,
        conv_state,
        x,
        conv1d_weight_,
        conv1d_bias_ ? std::optional<infinicore::Tensor>(conv1d_bias_) : std::nullopt,
        std::nullopt,
        std::nullopt,
        std::nullopt);
    conv_out = infinicore::op::silu(conv_out);

    auto ssm_params_in = conv_out;
    auto ssm_params = x_proj_->forward(ssm_params_in);
    auto time_step = ssm_params->narrow({{2, 0, time_step_rank_}})->contiguous();
    auto b = ssm_params->narrow({{2, time_step_rank_, state_size_}})->contiguous();
    auto c = ssm_params->narrow({{2, time_step_rank_ + state_size_, state_size_}})->contiguous();
    auto dt_in = time_step;
    auto dt = dt_proj_->forward(dt_in);
    auto scan = infinicore::op::mamba_selective_scan(conv_out, dt, b, c, A_log_, D_, gate, dt_bias_zero_, ssm_state);
    auto scan_mut = scan;
    return out_proj_->forward(scan_mut);
}

MambaBlock::MambaBlock(std::shared_ptr<infinilm::config::ModelConfig> config,
                       size_t layer_idx,
                       const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const double eps = config->get<double>("layer_norm_epsilon");
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mixer, config, layer_idx, device);
}

infinicore::Tensor MambaBlock::forward(const infinicore::Tensor &hidden_states) const {
    auto residual = hidden_states;
    auto x = norm_->forward(hidden_states);
    x = mixer_->forward(x);
    return infinicore::op::add(x, residual);
}

MambaModel::MambaModel(std::shared_ptr<infinilm::config::ModelConfig> config,
                       const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = config->get<size_t>("num_hidden_layers");
    const double eps = config->get<double>("layer_norm_epsilon");
    INFINICORE_NN_MODULE_INIT(embeddings, vocab_size, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<MambaBlock>("layers." + std::to_string(i), config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm_f, hidden_size, eps, dtype, device);
}

infinicore::Tensor MambaModel::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    if (input_ids->shape().size() == 1) {
        input_ids = input_ids->view({1, input_ids->shape()[0]});
    }
    auto hidden_states = embeddings_->forward(input_ids);
    for (const auto &layer : layers_) {
        hidden_states = layer->forward(hidden_states);
    }
    return norm_f_->forward(hidden_states);
}

MambaForCausalLM::MambaForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                                   const infinicore::Device &device) {
    model_config_ = config;
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t vocab_size = config->get<size_t>("vocab_size");
    INFINICORE_NN_MODULE_INIT(backbone, config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

InfinilmModel::Output MambaForCausalLM::forward(const InfinilmModel::Input &input) const {
    auto hidden_states = backbone_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MambaForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (nullptr == cache_config) {
        InfinilmModel::reset_cache(nullptr);
        return;
    }
    cache_config_ = cache_config->unique_copy();
    auto &cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
    cache_vec.clear();
    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");
    const size_t intermediate = model_config_->get<size_t>("intermediate_size");
    const size_t state_size = model_config_->get<size_t>("state_size");
    const size_t conv_kernel = model_config_->get<size_t>("conv_kernel");
    size_t max_batch_size = 1;
    if (auto static_config = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        max_batch_size = static_config->max_batch_size();
    } else if (auto paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(cache_config)) {
        max_batch_size = paged_config->num_blocks();
    }
    const auto &dtype = model_config_->get_dtype();
    const auto device = infinicore::context::getDevice();
    cache_vec.reserve(num_layers * 2);
    for (size_t i = 0; i < num_layers; ++i) {
        cache_vec.push_back(infinicore::Tensor::zeros({max_batch_size, intermediate, conv_kernel - 1}, dtype, device));
        cache_vec.push_back(infinicore::Tensor::zeros({max_batch_size, intermediate, state_size}, infinicore::DataType::F32, device));
    }
}

} // namespace infinilm::models::mamba

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    mamba,
    infinilm::models::mamba::MambaForCausalLM,
    infinilm::models::mamba::create_mamba_model_config);
} // namespace
