#include "glm4_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../utils.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"

#include <cmath>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::models::glm4 {

Glm4Attention::Glm4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                             size_t layer_idx,
                             const infinicore::Device &device)
    : model_config_(model_config),
      layer_idx_(layer_idx),
      hidden_size_(model_config->get<size_t>("hidden_size")),
      num_attention_heads_(model_config->get<size_t>("num_attention_heads")),
      num_key_value_heads_(model_config->get<size_t>("num_key_value_heads")),
      head_dim_(model_config->get_head_dim()),
      rotary_dim_(model_config->get_rotary_dim()),
      use_bias_(model_config->get_or<bool>("attention_bias", true)),
      use_output_bias_(model_config->get_or<bool>("attention_output_bias", false)) {

    const auto &dtype{model_config_->get_dtype()};

    const engine::distributed::RankInfo &g_rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    if ((num_key_value_heads_ >= tp_size) && (0 == (num_key_value_heads_ % tp_size))) {
        num_attention_heads_ /= tp_size;
        num_key_value_heads_ /= tp_size;
    } else {
        throw std::runtime_error("Glm4Attention: num_key_value_heads must be divisible by tp_size");
    }
    scaling_ = 1.0f / std::sqrt(static_cast<float>(head_dim_));

    // Linear layer initialization
    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, model_config_->get<size_t>("num_attention_heads"), model_config_->get<size_t>("num_key_value_heads"),
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias_, dtype, device, g_rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", model_config_->get<size_t>("num_attention_heads") * head_dim_, hidden_size_, quantization_method,
        use_output_bias_, dtype, device, tp_rank, tp_size, g_rank_info.comm);

    // RoPE initialization
    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
        num_attention_heads_, head_dim_, scaling_,
        num_key_value_heads_, layer_idx_,
        kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);

    // KV Cache quantization scale initialization
    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

infinicore::Tensor Glm4Attention::forward(const infinicore::Tensor &positions,
                                          infinicore::Tensor &hidden_states) {
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states);
    }
    return forward_paged_(positions, hidden_states);
}

infinicore::Tensor Glm4Attention::forward_paged_(const infinicore::Tensor &positions,
                                                 infinicore::Tensor &hidden_states) {
    if (!rotary_emb_) {
        throw std::runtime_error("Glm4Attention: rotary_emb not configured");
    }

    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // Paged mode strictly requires batch_size == 1 due to continuous batching scheduler flattening
    if (batch_size != 1) {
        throw std::runtime_error("Glm4Attention::forward_paged_ expects batch_size == 1");
    }

    // 1. QKV Projection
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states);

    // Reshape to 3D [sl, nh, hd] - Native layout required by Paged KV Cache update
    // DO NOT transpose K and V, otherwise do_kv_cache_update will cause segmentation fault!
    q = q->view({seq_len, num_attention_heads_, head_dim_});
    k = k->view({seq_len, num_key_value_heads_, head_dim_});
    v = v->view({seq_len, num_key_value_heads_, head_dim_});

    // 2. Position IDs
    infinicore::Tensor pos_ids_for_rope;
    if (positions->shape().size() == 2) {
        // Squeeze the batch dimension directly to 1D [seq_len]
        pos_ids_for_rope = positions->narrow({{0, 0, 1}})->view({seq_len});
    } else if (positions->shape().size() == 1) {
        pos_ids_for_rope = positions;
    } else {
        throw std::runtime_error("Unexpected position_ids shape in forward_paged_");
    }

    // 3. Rotary Position Embedding (RoPE)
    // Apply in-place on 3D tensors. Dimension is now 2 (hd) instead of 3 in 4D tensors.
    rotary_emb_->forward(q->narrow({{2, 0, rotary_dim_}}), pos_ids_for_rope, true);
    rotary_emb_->forward(k->narrow({{2, 0, rotary_dim_}}), pos_ids_for_rope, true);

    // 4. Attention computation
    // Pass 3D Q, K, V directly. PagedAttention backend handles KV cache updating internally.
    auto attn_output = attn_->forward(q, k, v);

    // 5. Output Projection
    // Reshape attention output to 2D [seq_len, hidden_size] for the linear projection
    auto output_2d = attn_output->view({seq_len, num_attention_heads_ * head_dim_});
    auto output = o_proj_->forward(output_2d);

    // Restore 3D [1, seq_len, hidden_size] to match the input hidden_states shape
    return output->view({1, seq_len, hidden_size_});
}

infinicore::Tensor Glm4Attention::forward_static_(const infinicore::Tensor &positions,
                                                  infinicore::Tensor &hidden_states) {
    if (!rotary_emb_) {
        throw std::runtime_error("Glm4Attention: rotary_emb not configured");
    }

    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t num_tokens = batch_size * seq_len;

    // 1. QKV Projection -> [bs, sl, nh, hd]
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states);
    q = q->contiguous()->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    k = k->contiguous()->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    v = v->contiguous()->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 2. Position IDs
    infinicore::Tensor pos_ids_for_rope;
    if (positions->shape().size() == 2) {
        pos_ids_for_rope = positions->narrow({{0, 0, 1}})->contiguous()->view({seq_len});
    } else if (positions->shape().size() == 1) {
        pos_ids_for_rope = positions->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 3. Rotary Position Embedding (RoPE)
    // Use `true` to perform in-place modification on non-contiguous narrow views
    rotary_emb_->forward(q->narrow({{3, 0, rotary_dim_}}), pos_ids_for_rope, true);
    rotary_emb_->forward(k->narrow({{3, 0, rotary_dim_}}), pos_ids_for_rope, true);

    // Trick: Create tensor with target physical layout [bs, nh, sl, hd],
    // permute to logical layout [bs, sl, nh, hd], and copy data.
    // This satisfies the Attention backend's memory format requirement without realigning data.
    auto q_in = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q->dtype(), q->device())
                    ->permute({0, 2, 1, 3});
    q_in->copy_from(q);

    auto k_in = infinicore::Tensor::empty({batch_size, num_key_value_heads_, seq_len, head_dim_}, k->dtype(), k->device())
                    ->permute({0, 2, 1, 3});
    k_in->copy_from(k);

    auto v_in = infinicore::Tensor::empty({batch_size, num_key_value_heads_, seq_len, head_dim_}, v->dtype(), v->device())
                    ->permute({0, 2, 1, 3});
    v_in->copy_from(v);

    // 4. Attention computation (q, k, v)
    auto attn_output = attn_->forward(q_in, k_in, v_in);

    // 5. Output Projection
    auto output_2d = attn_output->contiguous()->view({num_tokens, num_attention_heads_ * head_dim_});
    auto output = o_proj_->forward(output_2d);
    return output->view({batch_size, seq_len, hidden_size_});
}

} // namespace infinilm::models::glm4
