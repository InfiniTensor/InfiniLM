#include "rwkv5_for_causal_lm.hpp"
#include "../../global_state/global_state.hpp"
#include "../models_registry.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/layer_norm.hpp"
#include "infinicore/ops/lerp.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/relu.hpp"
#include "infinicore/ops/rwkv5_wkv.hpp"
#include "infinicore/ops/sigmoid.hpp"
#include "infinicore/ops/silu.hpp"

#include <stdexcept>
#include <string>

namespace infinilm::models::rwkv {

namespace {

infinicore::Tensor layer_state_view(const infinicore::Tensor &state,
                                    size_t batch_size,
                                    size_t layer_idx,
                                    size_t num_layers,
                                    size_t hidden_size) {
    return state->narrow({{0, 0, batch_size}, {1, layer_idx, 1}})
        ->as_strided({batch_size, hidden_size},
                     {static_cast<ptrdiff_t>(num_layers * hidden_size), 1});
}

infinicore::Tensor layer_wkv_state_view(const infinicore::Tensor &state,
                                        size_t batch_size,
                                        size_t layer_idx,
                                        size_t num_layers,
                                        size_t num_heads,
                                        size_t head_size) {
    const ptrdiff_t batch_stride = static_cast<ptrdiff_t>(num_layers * num_heads * head_size * head_size);
    return state->narrow({{0, 0, batch_size}, {1, layer_idx, 1}})
        ->as_strided({batch_size, num_heads, head_size, head_size},
                     {batch_stride,
                      static_cast<ptrdiff_t>(head_size * head_size),
                      static_cast<ptrdiff_t>(head_size),
                      1});
}

infinicore::Tensor scale_tensor(const infinicore::Tensor &x, float scale) {
    if (scale == 1.0f) {
        return x;
    }
    auto zeros = infinicore::Tensor::zeros(x->shape(), x->dtype(), x->device());
    return infinicore::op::lerp(zeros, x, scale);
}

} // namespace

std::shared_ptr<infinilm::config::ModelConfig> create_rwkv5_model_config(std::shared_ptr<infinilm::config::ModelConfig> model_config) {
    const std::string &model_type = model_config->get<std::string>("model_type");
    if ("rwkv5" != model_type) {
        throw std::runtime_error("infinilm::models::rwkv::create_rwkv5_model_config: model_type is not rwkv5");
    }
    auto &j = model_config->get_config_json();
    const size_t hidden_size = j.value("hidden_size", 2048);
    j["attention_hidden_size"] = j.value("attention_hidden_size", hidden_size);
    if (!j.contains("intermediate_size") || j["intermediate_size"].is_null()) {
        j["intermediate_size"] = static_cast<size_t>((static_cast<double>(hidden_size) * 3.5) / 32) * 32;
    }
    j["head_size"] = j.value("head_size", 64);
    j["num_attention_heads"] = j.value("num_attention_heads", j["attention_hidden_size"].get<size_t>() / j["head_size"].get<size_t>());
    j["head_size_divisor"] = j.value("head_size_divisor", 8);
    j["layer_norm_eps"] = j.value("layer_norm_eps", j.value("layer_norm_epsilon", 1e-5));
    j["max_position_embeddings"] = j.value("max_position_embeddings", j.value("context_length", 4096));
    if (!j.contains("dtype") && !j.contains("torch_dtype")) {
        j["torch_dtype"] = "bfloat16";
    }
    j["tie_word_embeddings"] = false;
    return model_config;
}

Rwkv5SelfAttention::Rwkv5SelfAttention(std::shared_ptr<infinilm::config::ModelConfig> config,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    hidden_size_ = config->get<size_t>("hidden_size");
    attention_hidden_size_ = config->get<size_t>("attention_hidden_size");
    head_size_ = config->get<size_t>("head_size");
    num_heads_ = attention_hidden_size_ / head_size_;
    head_size_divisor_ = config->get_or<size_t>("head_size_divisor", 8);
    auto quantization_method = config->get_quantization_method();

    INFINICORE_NN_PARAMETER_INIT(time_decay, ({num_heads_, head_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_faaaa, ({num_heads_, head_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_gate, ({1, 1, hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_key, ({1, 1, hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_value, ({1, 1, hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_receptance, ({1, 1, hidden_size_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(key, hidden_size_, attention_hidden_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(value, hidden_size_, attention_hidden_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(receptance, hidden_size_, attention_hidden_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(gate, hidden_size_, attention_hidden_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(output, attention_hidden_size_, hidden_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(ln_x, hidden_size_, 1e-5, dtype, device);
}

infinicore::Tensor Rwkv5SelfAttention::shifted_hidden_(const infinicore::Tensor &hidden_states,
                                                       infinicore::Tensor &state) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto shifted = infinicore::Tensor::empty(shape, hidden_states->dtype(), hidden_states->device());
    shifted->narrow({{1, 0, 1}})->view({batch_size, hidden_size_})->copy_from(state);
    for (size_t t = 1; t < seq_len; ++t) {
        shifted->narrow({{1, t, 1}})->copy_from(hidden_states->narrow({{1, t - 1, 1}}));
    }
    state->copy_from(hidden_states->narrow({{1, seq_len - 1, 1}})->view({batch_size, hidden_size_}));
    return shifted;
}

infinicore::Tensor Rwkv5SelfAttention::group_norm_(const infinicore::Tensor &x) const {
    const auto shape = x->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto zeros = infinicore::Tensor::zeros(shape, x->dtype(), x->device());
    auto scaled = infinicore::op::lerp(zeros, x, 1.0f / static_cast<float>(head_size_divisor_));
    auto y = infinicore::Tensor::empty(shape, x->dtype(), x->device());
    auto scaled_3d = scaled->view({batch_size * seq_len, num_heads_, head_size_});
    auto y_3d = y->view({batch_size * seq_len, num_heads_, head_size_});
    auto weight = ln_x_->weight();
    auto bias = ln_x_->bias();
    for (size_t h = 0; h < num_heads_; ++h) {
        auto x_h = scaled_3d->narrow({{1, h, 1}})->view({batch_size * seq_len, head_size_});
        auto y_h = y_3d->narrow({{1, h, 1}})->view({batch_size * seq_len, head_size_});
        auto w_h = weight->narrow({{0, h * head_size_, head_size_}});
        auto b_h = bias->narrow({{0, h * head_size_, head_size_}});
        infinicore::op::layer_norm_(y_h, x_h, w_h, b_h, 1e-5f);
    }
    return y;
}

infinicore::Tensor Rwkv5SelfAttention::forward(const infinicore::Tensor &hidden_states,
                                               infinicore::Tensor &attn_x_state,
                                               infinicore::Tensor &wkv_state) const {
    auto shifted = shifted_hidden_(hidden_states, attn_x_state);
    auto key_mix = infinicore::op::lerp(shifted, hidden_states, time_mix_key_);
    auto value_mix = infinicore::op::lerp(shifted, hidden_states, time_mix_value_);
    auto receptance_mix = infinicore::op::lerp(shifted, hidden_states, time_mix_receptance_);
    auto gate_mix = infinicore::op::lerp(shifted, hidden_states, time_mix_gate_);

    auto key_states = key_->forward(key_mix);
    auto value_states = value_->forward(value_mix);
    auto receptance_states = receptance_->forward(receptance_mix);
    auto gate_states = infinicore::op::silu(gate_->forward(gate_mix));
    auto wkv = infinicore::op::rwkv5_wkv(receptance_states, key_states, value_states, time_decay_, time_faaaa_, wkv_state);
    auto normed = group_norm_(wkv);
    auto gated = infinicore::op::mul(normed, gate_states);
    return output_->forward(gated);
}

Rwkv5FeedForward::Rwkv5FeedForward(std::shared_ptr<infinilm::config::ModelConfig> config,
                                   size_t layer_idx,
                                   const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    hidden_size_ = config->get<size_t>("hidden_size");
    intermediate_size_ = config->get<size_t>("intermediate_size");
    auto quantization_method = config->get_quantization_method();

    INFINICORE_NN_PARAMETER_INIT(time_mix_key, ({1, 1, hidden_size_}, dtype, device));
    INFINICORE_NN_PARAMETER_INIT(time_mix_receptance, ({1, 1, hidden_size_}, dtype, device));
    INFINICORE_NN_MODULE_INIT(key, hidden_size_, intermediate_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(receptance, hidden_size_, hidden_size_, quantization_method, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(value, intermediate_size_, hidden_size_, quantization_method, false, dtype, device);
}

infinicore::Tensor Rwkv5FeedForward::shifted_hidden_(const infinicore::Tensor &hidden_states,
                                                     infinicore::Tensor &state) const {
    const auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];
    auto shifted = infinicore::Tensor::empty(shape, hidden_states->dtype(), hidden_states->device());
    shifted->narrow({{1, 0, 1}})->view({batch_size, hidden_size_})->copy_from(state);
    for (size_t t = 1; t < seq_len; ++t) {
        shifted->narrow({{1, t, 1}})->copy_from(hidden_states->narrow({{1, t - 1, 1}}));
    }
    state->copy_from(hidden_states->narrow({{1, seq_len - 1, 1}})->view({batch_size, hidden_size_}));
    return shifted;
}

infinicore::Tensor Rwkv5FeedForward::forward(const infinicore::Tensor &hidden_states,
                                             infinicore::Tensor &ffn_x_state) const {
    auto shifted = shifted_hidden_(hidden_states, ffn_x_state);
    auto key_mix = infinicore::op::lerp(shifted, hidden_states, time_mix_key_);
    auto receptance_mix = infinicore::op::lerp(shifted, hidden_states, time_mix_receptance_);
    auto key_states = infinicore::op::relu(key_->forward(key_mix));
    key_states = infinicore::op::mul(key_states, key_states);
    auto value_states = value_->forward(key_states);
    auto receptance_states = infinicore::op::sigmoid(receptance_->forward(receptance_mix));
    return infinicore::op::mul(receptance_states, value_states);
}

Rwkv5Block::Rwkv5Block(std::shared_ptr<infinilm::config::ModelConfig> config,
                       size_t layer_idx,
                       const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const double eps = config->get<double>("layer_norm_eps");
    rescale_every_ = config->get_or<size_t>("rescale_every", 0);
    if (layer_idx_ == 0) {
        pre_ln_ = this->register_module<infinicore::nn::LayerNorm>("pre_ln", hidden_size, eps, dtype, device);
    }
    INFINICORE_NN_MODULE_INIT(ln1, hidden_size, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attention, config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(ln2, hidden_size, eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(feed_forward, config, layer_idx, device);
}

infinicore::Tensor Rwkv5Block::forward(const infinicore::Tensor &hidden_states,
                                       infinicore::Tensor &attn_x_state,
                                       infinicore::Tensor &wkv_state,
                                       infinicore::Tensor &ffn_x_state) const {
    auto x = hidden_states;
    if (pre_ln_) {
        x = pre_ln_->forward(x);
    }
    auto attn = attention_->forward(ln1_->forward(x), attn_x_state, wkv_state);
    if (rescale_every_ > 0) {
        const size_t scale_power = layer_idx_ / rescale_every_;
        if (scale_power > 0) {
            attn = scale_tensor(attn, 1.0f / static_cast<float>(size_t{1} << scale_power));
        }
    }
    x = infinicore::op::add(x, attn);
    auto ffn = feed_forward_->forward(ln2_->forward(x), ffn_x_state);
    if (rescale_every_ > 0) {
        const size_t scale_power = layer_idx_ / rescale_every_;
        if (scale_power > 0) {
            ffn = scale_tensor(ffn, 1.0f / static_cast<float>(size_t{1} << scale_power));
        }
    }
    return infinicore::op::add(x, ffn);
}

Rwkv5Model::Rwkv5Model(std::shared_ptr<infinilm::config::ModelConfig> config,
                       const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t num_hidden_layers = config->get<size_t>("num_hidden_layers");
    const double eps = config->get<double>("layer_norm_eps");
    rescale_every_ = config->get_or<size_t>("rescale_every", 0);

    INFINICORE_NN_MODULE_INIT(embeddings, vocab_size, hidden_size, std::nullopt, dtype, device);
    blocks_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        blocks_.push_back(this->register_module<Rwkv5Block>("blocks." + std::to_string(i), config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(ln_out, hidden_size, eps, dtype, device);
}

infinicore::Tensor Rwkv5Model::forward(const infinilm::InfinilmModel::Input &input,
                                       infinicore::Tensor &attn_x_state,
                                       infinicore::Tensor &wkv_state,
                                       infinicore::Tensor &ffn_x_state) const {
    auto hidden_states = embeddings_->forward(input.input_ids.value());
    const size_t batch_size = hidden_states->shape()[0];
    const size_t num_layers = blocks_.size();
    const size_t hidden_size = hidden_states->shape()[2];
    const size_t num_heads = wkv_state->shape()[2];
    const size_t head_size = wkv_state->shape()[3];

    for (size_t i = 0; i < num_layers; ++i) {
        auto attn_x = layer_state_view(attn_x_state, batch_size, i, num_layers, hidden_size);
        auto attn_kv = layer_wkv_state_view(wkv_state, batch_size, i, num_layers, num_heads, head_size);
        auto ffn_x = layer_state_view(ffn_x_state, batch_size, i, num_layers, hidden_size);
        hidden_states = blocks_.at(i)->forward(hidden_states, attn_x, attn_kv, ffn_x);
        if (rescale_every_ > 0 && (i + 1) % rescale_every_ == 0) {
            hidden_states = scale_tensor(hidden_states, 0.5f);
        }
    }
    return ln_out_->forward(hidden_states);
}

Rwkv5ForCausalLM::Rwkv5ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                                   const infinicore::Device &device)
    : device_(device), dtype_(config->get_dtype()) {
    model_config_ = config;
    num_hidden_layers_ = config->get<size_t>("num_hidden_layers");
    hidden_size_ = config->get<size_t>("hidden_size");
    head_size_ = config->get<size_t>("head_size");
    num_heads_ = config->get<size_t>("attention_hidden_size") / head_size_;
    const size_t vocab_size = config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(rwkv, config, device);
    INFINICORE_NN_MODULE_INIT(head, hidden_size_, vocab_size, false, dtype_, device);
}

void Rwkv5ForCausalLM::ensure_state_(size_t batch_size) const {
    if (state_batch_size_ >= batch_size && attn_x_state_ && wkv_state_ && ffn_x_state_) {
        return;
    }
    state_batch_size_ = batch_size;
    attn_x_state_ = infinicore::Tensor::zeros({batch_size, num_hidden_layers_, hidden_size_}, dtype_, device_);
    wkv_state_ = infinicore::Tensor::zeros({batch_size, num_hidden_layers_, num_heads_, head_size_, head_size_}, infinicore::DataType::F32, device_);
    ffn_x_state_ = infinicore::Tensor::zeros({batch_size, num_hidden_layers_, hidden_size_}, dtype_, device_);
}

InfinilmModel::Output Rwkv5ForCausalLM::forward(const InfinilmModel::Input &input) const {
    const size_t batch_size = input.input_ids.value()->shape()[0];
    ensure_state_(batch_size);
    auto hidden_states = rwkv_->forward(input, attn_x_state_, wkv_state_, ffn_x_state_);
    auto logits = head_->forward(hidden_states);
    return {logits};
}

void Rwkv5ForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        cache_config_.reset();
    } else {
        cache_config_ = cache_config->unique_copy();
    }
    infinilm::global_state::get_forward_context().kv_cache_vec.clear();
    state_batch_size_ = 0;
    attn_x_state_.reset();
    wkv_state_.reset();
    ffn_x_state_.reset();
}

} // namespace infinilm::models::rwkv

namespace {
INFINILM_REGISTER_CAUSAL_LM_MODEL(
    rwkv5,
    infinilm::models::rwkv::Rwkv5ForCausalLM,
    infinilm::models::rwkv::create_rwkv5_model_config);
} // namespace
