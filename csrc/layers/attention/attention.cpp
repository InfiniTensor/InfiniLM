#include "attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../rotary_embedding/rotary_embedding.hpp"
#include <string>

namespace infinilm::layers::attention {

Attention::Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t layer_idx,
                     const infinicore::Device &device)
    : device_(device),
      dtype_(model_config->get_dtype()) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", true);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < tp_size ? 1 : total_num_kv_heads / tp_size;

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype_, device_, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype_, device_, tp_rank, tp_size, rank_info.comm);

    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device_);

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    attn_ = std::make_shared<AttentionLayer>(num_attention_heads_, head_dim_, scaling, num_key_value_heads_, layer_idx_,
                                             kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_, device_);

    init_kv_cache_quant_params(register_fn, device_, kv_cache_k_scale_, kv_cache_v_scale_);

    rank_qkv_output_size_ = qkv_proj_->out_features() / static_cast<size_t>(tp_size);
    this->_initialize_preallocated_workspace();
}

infinicore::Tensor Attention::forward(const infinicore::Tensor &positions,
                                      const infinicore::Tensor &hidden_states) const {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Attention::forward_static_(const infinicore::Tensor &position_ids,
                                              const infinicore::Tensor &hidden_states) const {
    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto qkv_output = max_qkv_output_->narrow({{0, 0, batch_size * seq_len}})->view({batch_size, seq_len, rank_qkv_output_size_});
    auto [q, k, v] = qkv_proj_->forward_split_(qkv_output, hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("infinilm::layers::attention::Attention: Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_rope, k_reshaped, v_reshaped);

    // 7. Project output
    auto o_output = max_o_output_->narrow({{0, 0, batch_size * seq_len}})->view({batch_size, seq_len, hidden_size_});
    o_proj_->forward_(o_output, attn_output);
    return o_output;
}

infinicore::Tensor Attention::forward_paged_(const infinicore::Tensor &position_ids,
                                             const infinicore::Tensor &hidden_states) const {
    // hidden_states shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Only support batchsize==1, all requests should be flattened along seqlen dimension
    ASSERT_EQ(batch_size, 1);

    // 1. Project Q, K, V
    auto qkv_output = max_qkv_output_->narrow({{0, 0, seq_len}})->view({1, seq_len, rank_qkv_output_size_});
    auto [q, k, v] = qkv_proj_->forward_split_(qkv_output, hidden_states_mutable);

    // 2. Reshape for multi-head attention
    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids;
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to QK
    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);

    // 5. Attn Backend calculate
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);

    // 6. Project output
    auto o_output = max_o_output_->narrow({{0, 0, seq_len}})->view({1, seq_len, hidden_size_});
    o_proj_->forward_(o_output, attn_output);
    return o_output;
}

void Attention::_initialize_preallocated_workspace() {
    const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
    auto &preallocated_workspace = infinilm::global_state::get_forward_context().preallocated_workspace;
    const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

    const std::string attention_cache_key = std::string("Attention_max_num_batched_tokens_")
                                          + std::to_string(max_num_batched_tokens) + "_rank_qkv_output_size_"
                                          + std::to_string(rank_qkv_output_size_) + "_hidden_size_"
                                          + std::to_string(hidden_size_) + "_dtype_"
                                          + infinicore::toString(dtype_) + "_device_"
                                          + device_.toString();

    size_t max_output_size = std::max(rank_qkv_output_size_, hidden_size_);
    if (preallocated_workspace.find(attention_cache_key) == preallocated_workspace.end()) {
        auto attention_buffer = infinicore::Tensor::empty({max_num_batched_tokens * max_output_size}, dtype_, device_);
        preallocated_workspace[attention_cache_key] = attention_buffer;
    }

    auto attention_buffer = preallocated_workspace.at(attention_cache_key);
    const auto attention_buffer_shape = attention_buffer->shape();
    ASSERT(attention_buffer_shape[0] == max_num_batched_tokens * max_output_size);

    max_qkv_output_ = attention_buffer->narrow({{0, 0, max_num_batched_tokens * rank_qkv_output_size_}})->view({max_num_batched_tokens, rank_qkv_output_size_});
    max_o_output_ = attention_buffer->narrow({{0, 0, max_num_batched_tokens * hidden_size_}})->view({max_num_batched_tokens, hidden_size_});
}

void init_kv_cache_quant_params(std::function<void(const std::string &, infinicore::nn::Parameter)> register_fn,
                                const infinicore::Device &device,
                                infinicore::nn::Parameter &kv_cache_k_scale,
                                infinicore::nn::Parameter &kv_cache_v_scale) {
    auto kv_quant_scheme = infinilm::global_state::get_infinilm_config().model_config->get_kv_quant_scheme();
    switch (kv_quant_scheme) {
    case infinilm::quantization::KVQuantAlgo::NONE:
        break;
    case infinilm::quantization::KVQuantAlgo::INT8:
        kv_cache_k_scale = infinicore::nn::Parameter({1}, infinicore::DataType::F32, device, 0, 0, 1);
        register_fn("kv_cache_k_scale", kv_cache_k_scale);
        kv_cache_v_scale = infinicore::nn::Parameter({1}, infinicore::DataType::F32, device, 0, 0, 1);
        register_fn("kv_cache_v_scale", kv_cache_v_scale);
        break;
    default:
        throw std::runtime_error("unsupported kv_quant_scheme");
    }
}

} // namespace infinilm::layers::attention
