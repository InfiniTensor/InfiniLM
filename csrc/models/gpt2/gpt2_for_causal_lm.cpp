#include "gpt2_for_causal_lm.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/attention/attention.hpp"
#include "../models_registry.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::gpt2 {

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
    j["num_key_value_heads"] = j["num_attention_heads"];
    j["head_dim"] = j["hidden_size"].get<size_t>() / j["num_attention_heads"].get<size_t>();
    j["max_position_embeddings"] = j.value("max_position_embeddings", j.value("n_positions", 1024));
    j["intermediate_size"] = j.value("n_inner", 4 * j["hidden_size"].get<size_t>());
    j["layer_norm_eps"] = j.value("layer_norm_epsilon", 1e-5);
    j["attention_bias"] = true;
    j["attention_output_bias"] = false;
    j["mlp_bias"] = true;

    return config;
}

GPT2Attention::GPT2Attention(std::shared_ptr<infinilm::config::ModelConfig> config,
                             size_t layer_idx,
                             const infinicore::Device &device)
    : layer_idx_(layer_idx) {
    const auto &dtype = config->get_dtype();
    hidden_size_ = config->get<size_t>("hidden_size");
    num_heads_ = config->get<size_t>("num_attention_heads");
    num_kv_heads_ = config->get<size_t>("num_key_value_heads");
    head_dim_ = config->get<size_t>("head_dim");

    const bool use_bias = config->get_or<bool>("attention_bias", true);
    auto quantization_method = config->get_quantization_method();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    const int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    const size_t total_num_heads = num_heads_;
    const size_t total_num_kv_heads = num_kv_heads_;

    num_heads_ = total_num_heads / tp_size;
    num_kv_heads_ = total_num_kv_heads < static_cast<size_t>(tp_size)
        ? 1
        : total_num_kv_heads / tp_size;

    auto register_fn = [this](const std::string &name, infinicore::nn::Parameter param) {
        this->register_parameter(name, std::move(param));
    };
    qkv_proj_ = std::make_shared<infinilm::layers::linear::QKVParallelLinear>(
        hidden_size_,
        head_dim_,
        total_num_heads,
        total_num_kv_heads,
        "q_proj",
        "k_proj",
        "v_proj",
        register_fn,
        quantization_method,
        use_bias,
        dtype,
        device,
        rank_info);
    INFINICORE_NN_MODULE_INIT(
        o_proj,
        total_num_heads * head_dim_,
        hidden_size_,
        quantization_method,
        false,
        dtype,
        device,
        tp_rank,
        tp_size,
        rank_info.comm);
    INFINICORE_NN_PARAMETER_INIT(o_proj_bias, ({hidden_size_}, dtype, device));

    infinilm::layers::attention::init_kv_cache_quant_params(
        register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_heads_,
        head_dim_,
        scale,
        num_kv_heads_,
        layer_idx_,
        kv_cache_k_scale_,
        kv_cache_v_scale_,
        attention_backend_);
}

infinicore::Tensor GPT2Attention::forward(const infinicore::Tensor &positions,
                                          const infinicore::Tensor &hidden_states) const {
    (void)positions;
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    if (attention_backend_ == infinilm::backends::AttentionBackend::PAGED_ATTN
        || attention_backend_ == infinilm::backends::AttentionBackend::FLASH_ATTN) {
        auto q_reshaped = q->view({seq_len, num_heads_, head_dim_});
        auto k_reshaped = k->view({seq_len, num_kv_heads_, head_dim_});
        auto v_reshaped = v->view({seq_len, num_kv_heads_, head_dim_});
        auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
        auto output = o_proj_->forward(attn_output);
        infinicore::op::add_(output, output, o_proj_bias_->as_strided(output->shape(), {0, 0, 1}));
        return output;
    }

    auto q_reshaped = q->view({batch_size, seq_len, num_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_kv_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_kv_heads_, head_dim_});
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    auto output = o_proj_->forward(attn_output);
    infinicore::op::add_(output, output, o_proj_bias_->as_strided(output->shape(), {0, 0, 1}));
    return output;
}

GPT2MLP::GPT2MLP(std::shared_ptr<infinilm::config::ModelConfig> config,
                 const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t intermediate_size = config->get<size_t>("intermediate_size");
    const bool use_bias = config->get_or<bool>("mlp_bias", true);
    activation_ = config->get_or<std::string>("activation_function", "gelu_new");
    auto quantization_method = config->get_quantization_method();
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();

    INFINICORE_NN_MODULE_INIT(
        c_fc,
        hidden_size,
        intermediate_size,
        quantization_method,
        use_bias,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size);
    INFINICORE_NN_MODULE_INIT(
        c_proj,
        intermediate_size,
        hidden_size,
        quantization_method,
        false,
        dtype,
        device,
        rank_info.tp_rank,
        rank_info.tp_size,
        rank_info.comm);
    INFINICORE_NN_PARAMETER_INIT(c_proj_bias, ({hidden_size}, dtype, device));
}

infinicore::Tensor GPT2MLP::forward(const infinicore::Tensor &hidden_states) const {
    auto x = const_cast<infinicore::Tensor &>(hidden_states);
    x = c_fc_->forward(x);
    if (activation_ == "gelu_new" || activation_ == "gelu_tanh") {
        x = infinicore::op::gelu_tanh(x);
    } else if (activation_ == "gelu") {
        x = infinicore::op::gelu(x);
    } else {
        throw std::runtime_error("infinilm::models::gpt2::GPT2MLP: unsupported activation " + activation_);
    }
    x = c_proj_->forward(x);
    infinicore::op::add_(x, x, c_proj_bias_->as_strided(x->shape(), {0, 0, 1}));
    return x;
}

GPT2Block::GPT2Block(std::shared_ptr<infinilm::config::ModelConfig> config,
                     size_t layer_idx,
                     const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const double layer_norm_eps = config->get<double>("layer_norm_eps");

    INFINICORE_NN_MODULE_INIT(ln_1, hidden_size, layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(attn, config, layer_idx, device);
    INFINICORE_NN_MODULE_INIT(ln_2, hidden_size, layer_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, device);
}

infinicore::Tensor GPT2Block::forward(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const {
    auto residual = hidden_states;
    auto x = ln_1_->forward(hidden_states);
    x = attn_->forward(positions, x);
    x = infinicore::op::add(x, residual);

    residual = x;
    x = ln_2_->forward(x);
    x = mlp_->forward(x);
    return infinicore::op::add(x, residual);
}

GPT2Model::GPT2Model(std::shared_ptr<infinilm::config::ModelConfig> config,
                     const infinicore::Device &device) {
    const auto &dtype = config->get_dtype();
    const size_t vocab_size = config->get<size_t>("vocab_size");
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t max_position_embeddings = config->get<size_t>("max_position_embeddings");
    const size_t num_hidden_layers = config->get<size_t>("num_hidden_layers");
    const double layer_norm_eps = config->get<double>("layer_norm_eps");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_INIT(embed_positions, max_position_embeddings, hidden_size, std::nullopt, dtype, device);
    layers_.reserve(num_hidden_layers);
    for (size_t i = 0; i < num_hidden_layers; ++i) {
        layers_.push_back(this->register_module<GPT2Block>("layers." + std::to_string(i), config, i, device));
    }
    INFINICORE_NN_MODULE_INIT(norm, hidden_size, layer_norm_eps, dtype, device);
}

infinicore::Tensor GPT2Model::forward(const infinilm::InfinilmModel::Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();
    if (position_ids->shape().size() == 1) {
        position_ids = position_ids->view({1, position_ids->shape()[0]});
    }

    auto hidden_states = infinicore::op::add(
        embed_tokens_->forward(input_ids),
        embed_positions_->forward(position_ids));

    for (const auto &layer : layers_) {
        hidden_states = layer->forward(position_ids, hidden_states);
    }

    return norm_->forward(hidden_states);
}

GPT2ForCausalLM::GPT2ForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> config,
                                 const infinicore::Device &device) {
    model_config_ = config;
    const auto &dtype = config->get_dtype();
    const size_t hidden_size = config->get<size_t>("hidden_size");
    const size_t vocab_size = config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(model, config, device);
    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

InfinilmModel::Output GPT2ForCausalLM::forward(const InfinilmModel::Input &input) const {
    auto hidden_states = model_->forward(input);
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

} // namespace infinilm::models::gpt2

namespace {

INFINILM_REGISTER_CAUSAL_LM_MODEL(
    gpt2,
    infinilm::models::gpt2::GPT2ForCausalLM,
    infinilm::models::gpt2::create_gpt2_model_config);

} // namespace
