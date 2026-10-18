#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/embedding/embedding.hpp"
#include "../../layers/normalization/layer_norm.hpp"
#include "gpt2_for_causal_lm.hpp"
#include "../../global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "../../layers/attention/attention.hpp"
#include "../models_registry.hpp"
#include <cmath>
#include <exception>
#include <limits>

namespace infinilm::models::gpt2 {
namespace {
void validate_gpt2_resources(infinilm::runtime::Runtime &runtime,
                             const std::shared_ptr<config::ModelConfig> &config,
                             const engine::distributed::RankInfo &rank) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("GPT2 construction requires eager Runtime");
    if (!config || rank.device != runtime.device() || rank.tp_size <= 0
        || rank.tp_rank < 0 || rank.tp_rank >= rank.tp_size || (rank.tp_size > 1 && !rank.comm))
        throw std::invalid_argument("Invalid explicit GPT2 construction resources");
}
} // namespace

std::shared_ptr<infinilm::config::ModelConfig>
create_gpt2_model_config(std::shared_ptr<infinilm::config::ModelConfig> config) {
    const std::string &model_type = config->get<std::string>("model_type");
    if ("gpt2" != model_type) {
        throw std::runtime_error(
            "infinilm::models::gpt2::create_gpt2_model_config: model_type is not gpt2");
    }

    auto &j = config->get_config_json();

    j["hidden_size"] = j.value("hidden_size", j.value("n_embd", 768));
    j["num_hidden_layers"] = j.value("num_hidden_layers", j.value("n_layer", 12));
    j["num_attention_heads"] = j.value("num_attention_heads", j.value("n_head", 12));
    const auto hidden = config->get<int64_t>("hidden_size");
    const auto heads = config->get<int64_t>("num_attention_heads");
    if (hidden <= 0 || heads <= 0 || hidden % heads != 0 || config->get<int64_t>("num_hidden_layers") < 0)
        throw std::invalid_argument("GPT2 requires valid hidden/head/layer dimensions");
    if (static_cast<size_t>(hidden) > std::numeric_limits<size_t>::max() / 4)
        throw std::length_error("GPT2 default intermediate size overflow");
    j["num_key_value_heads"] = j["num_attention_heads"];
    j["head_dim"] = j["hidden_size"].get<size_t>() / j["num_attention_heads"].get<size_t>();
    j["max_position_embeddings"] = j.value("max_position_embeddings", j.value("n_positions", 1024));
    j["intermediate_size"] = !j.contains("n_inner") || j.at("n_inner").is_null()
        ? nlohmann::json(4 * j["hidden_size"].get<size_t>()) : j.at("n_inner");
    j["layer_norm_eps"] = j.value("layer_norm_epsilon", 1e-5);
    j["attention_bias"] = true;
    j["attention_output_bias"] = false;
    j["mlp_bias"] = true;
    for (const char *key : {"vocab_size", "max_position_embeddings", "intermediate_size"})
        if (config->get<int64_t>(key) <= 0) throw std::invalid_argument(std::string("GPT2 requires positive ") + key);
    const double epsilon = config->get<double>("layer_norm_eps");
    if (!std::isfinite(epsilon) || epsilon <= 0) throw std::invalid_argument("GPT2 requires valid LayerNorm epsilon");
    j["activation_function"] = j.value("activation_function", std::string("gelu_new"));
    const auto activation = config->get<std::string>("activation_function");
    if (activation != "gelu_new" && activation != "gelu_tanh" && activation != "gelu")
        throw std::invalid_argument("GPT2 does not implement activation " + activation);
    if (j.value("add_cross_attention", false))
        throw std::invalid_argument("GPT2 does not implement add_cross_attention");
    if (!j.value("scale_attn_weights", true))
        throw std::invalid_argument("GPT2 requires scale_attn_weights=true");
    if (j.value("scale_attn_by_inverse_layer_idx", false))
        throw std::invalid_argument("GPT2 does not implement scale_attn_by_inverse_layer_idx");
    if (j.value("reorder_and_upcast_attn", false))
        throw std::invalid_argument("GPT2 does not implement reorder_and_upcast_attn");

    return config;
}

GPT2Attention::GPT2Attention(infinilm::runtime::Runtime &runtime,
                             std::shared_ptr<infinilm::config::ModelConfig> config,
                             size_t layer_idx, const engine::distributed::RankInfo &rank_info,
                             backends::AttentionBackend backend, const layers::moe::EPConfig &ep)
    : layer_idx_(layer_idx) {
    validate_gpt2_resources(runtime, config, rank_info);
    (void)ep;
    if (backend != backends::AttentionBackend::STATIC_ATTN
        && backend != backends::AttentionBackend::PAGED_ATTN
        && backend != backends::AttentionBackend::FLASH_ATTN)
        throw std::invalid_argument("Unsupported explicit GPT2 Attention backend");
    const auto &dtype = config->get_dtype();
    hidden_size_ = config->get<size_t>("hidden_size");
    num_heads_ = config->get<size_t>("num_attention_heads");
    num_kv_heads_ = config->get<size_t>("num_key_value_heads");
    head_dim_ = config->get<size_t>("head_dim");

    const bool use_bias = config->get_or<bool>("attention_bias", true);
    auto quantization_method = config->get_quantization_method();
    const int tp_rank = rank_info.tp_rank;
    const int tp_size = rank_info.tp_size;
    const size_t total_num_heads = num_heads_;
    const size_t total_num_kv_heads = num_kv_heads_;

    num_heads_ = total_num_heads / tp_size;
    num_kv_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
        ? 1
        : total_num_kv_heads / tp_size;

    auto register_fn = [this](const std::string &name, infinilm::weights::Weight param) {
        this->bind_weight(name, std::move(param));
    };
    try {
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
            runtime, hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
            "q_proj", "k_proj", "v_proj", register_fn, quantization_method, use_bias, dtype, rank_info);
        o_proj_ = std::make_shared<infinilm::layers::linear::RowParallelLinear>(
            runtime, total_num_heads * head_dim_, hidden_size_, quantization_method,
            false, dtype, tp_rank, tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);
        o_proj_bias_ = infinilm::weights::Weight(runtime, {hidden_size_}, dtype);
        this->bind_weight("o_proj_bias", o_proj_bias_);

    infinilm::layers::attention::init_kv_cache_quant_params(
            runtime, register_fn, kv_cache_k_scale_, kv_cache_v_scale_, config->get_kv_quant_scheme());

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        attention_backend_ = backend;
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
            num_heads_, head_dim_, scale, num_kv_heads_, layer_idx_, kv_cache_k_scale_, kv_cache_v_scale_,
            attention_backend_, config->get_kv_quant_scheme());
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor GPT2Attention::forward(const infinilm::runtime::Tensor &positions,
                                          const infinilm::runtime::Tensor &hidden_states,
                                          infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    (void)positions;
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(runtime, hidden_states_mutable);

    if (attention_backend_ == infinilm::backends::AttentionBackend::PAGED_ATTN
        || attention_backend_ == infinilm::backends::AttentionBackend::FLASH_ATTN) {
        auto q_reshaped = q->view({seq_len, num_heads_, head_dim_});
        auto k_reshaped = k->view({seq_len, num_kv_heads_, head_dim_});
        auto v_reshaped = v->view({seq_len, num_kv_heads_, head_dim_});
        auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);
        auto output = o_proj_->forward(runtime, attn_output);
        {
            const auto ops_other = o_proj_bias_->as_strided(output->shape(), {0, 0, 1});
            infinilm::execution::submit<infini::ops::Add>(
                runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), output, ops_other, output);
            INFINILM_TRACE_OP(Add, output);
        }
        return output;
    }

    auto q_reshaped = q->view({batch_size, seq_len, num_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_kv_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_kv_heads_, head_dim_});
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);
    auto output = o_proj_->forward(runtime, attn_output);
    {
        const auto ops_other = o_proj_bias_->as_strided(output->shape(), {0, 0, 1});
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), output, ops_other, output);
        INFINILM_TRACE_OP(Add, output);
    }
    return output;
}

GPT2MLP::GPT2MLP(infinilm::runtime::Runtime &runtime,
                 std::shared_ptr<infinilm::config::ModelConfig> config,
                 const engine::distributed::RankInfo &rank_info) {
    validate_gpt2_resources(runtime, config, rank_info);
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t intermediate_size = config->get<size_t>("intermediate_size");
    const bool use_bias = config->get_or<bool>("mlp_bias", true);
    activation_ = config->get_or<std::string>("activation_function", "gelu_new");
    auto quantization_method = config->get_quantization_method();

    try {
        c_fc_ = std::make_shared<infinilm::layers::linear::ColumnParallelLinear>(
            runtime, hidden_size, intermediate_size, quantization_method, use_bias, dtype,
            rank_info.tp_rank, rank_info.tp_size);
        this->bind_child("c_fc", c_fc_);
        c_proj_ = std::make_shared<infinilm::layers::linear::RowParallelLinear>(
            runtime, intermediate_size, hidden_size, quantization_method, false, dtype,
            rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("c_proj", c_proj_);
        c_proj_bias_ = infinilm::weights::Weight(runtime, {hidden_size}, dtype);
        this->bind_weight("c_proj_bias", c_proj_bias_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor GPT2MLP::forward(infinilm::runtime::Runtime &runtime,
                                   const infinilm::runtime::Tensor &hidden_states) const {
    runtime.assertOwnerThread();
    auto x = hidden_states;
    x = c_fc_->forward(runtime, x);
    if (activation_ == "gelu_new" || activation_ == "gelu_tanh") {
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"tanh"}, ops_output);
            x = std::move(ops_output);
        }
    } else if (activation_ == "gelu") {
        {
            auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
            infinilm::execution::submit<infini::ops::Gelu>(
                runtime, infini::ops::Gelu::DefaultConfig(runtime.device().type()), x, std::string{"none"}, ops_output);
            x = std::move(ops_output);
        }
    } else {
        throw std::runtime_error("infinilm::models::gpt2::GPT2MLP: unsupported activation " + activation_);
    }
    x = c_proj_->forward(runtime, x);
    {
        const auto ops_other = c_proj_bias_->as_strided(x->shape(), {0, 0, 1});
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, ops_other, x);
        INFINILM_TRACE_OP(Add, x);
    }
    return x;
}

GPT2Block::GPT2Block(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> config,
                     size_t layer_idx, const engine::distributed::RankInfo &rank,
                     backends::AttentionBackend backend, const layers::moe::EPConfig &ep) {
    validate_gpt2_resources(runtime, config, rank);
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const double layer_norm_eps = config->get<double>("layer_norm_eps");

    try {
        ln_1_ = std::make_shared<infinilm::layers::LayerNorm>(runtime, hidden_size, layer_norm_eps, dtype);
        this->bind_child("ln_1", ln_1_);
        attn_ = std::make_shared<GPT2Attention>(runtime, config, layer_idx, rank, backend, ep);
        this->bind_child("attn", attn_);
        ln_2_ = std::make_shared<infinilm::layers::LayerNorm>(runtime, hidden_size, layer_norm_eps, dtype);
        this->bind_child("ln_2", ln_2_);
        mlp_ = std::make_shared<GPT2MLP>(runtime, config, rank);
        this->bind_child("mlp", mlp_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor GPT2Block::forward(const infinilm::runtime::Tensor &positions,
                                      const infinilm::runtime::Tensor &hidden_states,
                                      infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto residual = hidden_states;
    auto x = ln_1_->forward(runtime, hidden_states);
    x = attn_->forward(positions, x, forward_context);
    {
        auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
        infinilm::execution::submit<infini::ops::Add>(
            runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
        INFINILM_TRACE_OP(Add, ops_output);
        x = std::move(ops_output);
    }

    residual = x;
    x = ln_2_->forward(runtime, x);
    x = mlp_->forward(runtime, x);
    auto ops_output = infinilm::runtime::Tensor::empty(runtime, x->shape(), x->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), x, residual, ops_output);
    INFINILM_TRACE_OP(Add, ops_output);
    return ops_output;
}

GPT2Model::GPT2Model(infinilm::runtime::Runtime &runtime,
                     std::shared_ptr<infinilm::config::ModelConfig> config,
                     const engine::distributed::RankInfo &rank,
                     backends::AttentionBackend backend, const layers::moe::EPConfig &ep) {
    validate_gpt2_resources(runtime, config, rank);
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t max_position_embeddings = config->get<size_t>("max_position_embeddings");
    const size_t num_hidden_layers = config->get<size_t>("num_hidden_layers");
    const double layer_norm_eps = config->get<double>("layer_norm_eps");

    try {
        embed_tokens_ = std::make_shared<infinilm::layers::Embedding>(runtime, vocab_size, hidden_size, std::nullopt, dtype);
        this->bind_child("embed_tokens", embed_tokens_);
        embed_positions_ = std::make_shared<infinilm::layers::Embedding>(runtime, max_position_embeddings, hidden_size, std::nullopt, dtype);
        this->bind_child("embed_positions", embed_positions_);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(std::make_shared<GPT2Block>(runtime, config, i, rank, backend, ep));
            this->bind_child("layers." + std::to_string(i), layers_.back());
    }
        norm_ = std::make_shared<infinilm::layers::LayerNorm>(runtime, hidden_size, layer_norm_eps, dtype);
        this->bind_child("norm", norm_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

infinilm::runtime::Tensor GPT2Model::forward(const infinilm::InfinilmModel::Input &input,
                                     infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();
    if (position_ids->shape().size() == 1) {
        position_ids = position_ids->view({1, position_ids->shape()[0]});
    }

    const auto ops_input = embed_tokens_->forward(runtime, input_ids);
    const auto ops_other = embed_positions_->forward(runtime, position_ids);
    auto hidden_states = infinilm::runtime::Tensor::empty(runtime, ops_input->shape(), ops_input->dtype());
    infinilm::execution::submit<infini::ops::Add>(
        runtime, infini::ops::Add::DefaultConfig(runtime.device().type()), ops_input, ops_other, hidden_states);
    INFINILM_TRACE_OP(Add, hidden_states);

    for (const auto &layer : layers_) {
        hidden_states = layer->forward(position_ids, hidden_states, forward_context);
    }

    return norm_->forward(runtime, hidden_states);
}

GPT2ForCausalLM::GPT2ForCausalLM(infinilm::runtime::Runtime &runtime,
                                 std::shared_ptr<infinilm::config::ModelConfig> config,
                                 const engine::distributed::RankInfo &rank,
                                 backends::AttentionBackend backend, const layers::moe::EPConfig &ep) {
    validate_runtime_construction(config, runtime, rank);
    model_config_ = config;
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t vocab_size = config->get<size_t>("vocab_size");

    try {
        model_ = std::make_shared<GPT2Model>(runtime, config, rank, backend, ep);
        this->bind_child("model", model_);
        lm_head_ = std::make_shared<infinilm::layers::linear::ReplicatedLinear>(runtime, hidden_size, vocab_size, false, dtype);
        this->bind_child("lm_head", lm_head_);
    } catch (...) {
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        throw;
    }
}

InfinilmModel::Output GPT2ForCausalLM::forward(
    const InfinilmModel::Input &input, infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    auto hidden_states = model_->forward(input, forward_context);
    auto logits = lm_head_->forward(runtime, hidden_states);
    return {logits, hidden_states};
}

} // namespace infinilm::models::gpt2

namespace {

INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(
    gpt2,
    infinilm::models::gpt2::GPT2ForCausalLM,
    infinilm::models::gpt2::create_gpt2_model_config);

} // namespace
