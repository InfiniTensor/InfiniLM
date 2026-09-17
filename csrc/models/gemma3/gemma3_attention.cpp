#include "gemma3_attention.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/quantization/quantization_scheme.hpp"
#include "../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/broadcast_to.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/mul_scalar.hpp"
#include <cstring>

namespace infinilm::models::gemma3 {

namespace {
constexpr float kMaskValue = -1e9f;

size_t read_first_len_(const infinicore::Tensor &lengths) {
    return static_cast<size_t>(
        reinterpret_cast<int32_t *>(lengths->to(infinicore::Device::cpu())->data())[0]);
}
} // namespace

Gemma3Attention::Gemma3Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 size_t layer_idx,
                                 const infinicore::Device &device) {
    layer_idx_ = layer_idx;
    hidden_size_ = model_config->get<size_t>("hidden_size");
    head_dim_ = model_config->get<size_t>("head_dim");

    const auto &dtype{model_config->get_dtype()};
    size_t total_num_heads = model_config->get<size_t>("num_attention_heads");
    size_t total_num_kv_heads = model_config->get<size_t>("num_key_value_heads");
    bool use_bias = model_config->get_or<bool>("attention_bias", false);
    bool use_output_bias = model_config->get_or<bool>("attention_output_bias", false);
    double rms_norm_eps = model_config->get<double>("rms_norm_eps");

    attention_backend_ = infinilm::global_state::get_infinilm_config().attention_backend;
    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();

    num_attention_heads_ = total_num_heads / tp_size;
    num_key_value_heads_ = total_num_kv_heads < tp_size ? 1 : total_num_kv_heads / tp_size;

    // Gemma-3 alternates sliding (local) and full (global) attention layers.
    const auto &layer_types = model_config->get_ref("layer_types");
    is_sliding_ = layer_types[layer_idx].get<std::string>() == "sliding_attention";
    sliding_window_ = model_config->get_or<size_t>("sliding_window", 0);
    if (is_sliding_ && sliding_window_ == 0) {
        throw std::runtime_error(
            "infinilm::models::gemma3::Gemma3Attention: layer_types marks this layer as sliding_attention "
            "but the config has no positive sliding_window");
    }

    auto quantization_method = model_config->get_quantization_method();
    auto register_fn = [this](const std::string &n, infinicore::nn::Parameter p) { this->register_parameter(n, std::move(p)); };
    qkv_proj_ = std::make_shared<layers::linear::QKVParallelLinear>(
        hidden_size_, head_dim_, total_num_heads, total_num_kv_heads,
        "q_proj", "k_proj", "v_proj", register_fn,
        quantization_method, use_bias, dtype, device, rank_info);
    o_proj_ = this->register_module<layers::linear::RowParallelLinear>(
        "o_proj", total_num_heads * head_dim_, hidden_size_, quantization_method,
        use_output_bias, dtype, device, tp_rank, tp_size, rank_info.comm);

    INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, rms_norm_eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, rms_norm_eps, dtype, device);

    // Sliding layers use the local RoPE base frequency, full layers the global one.
    double theta = is_sliding_ ? model_config->get_or<double>("rope_local_base_freq", 10000.0)
                               : model_config->get<double>("rope_theta");
    size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(
        head_dim_, head_dim_, max_position_embeddings, theta,
        model_config->get_rope_algo(), dtype, device, nullptr);

    float query_pre_attn_scalar = model_config->get_or<float>("query_pre_attn_scalar", static_cast<float>(head_dim_));
    scale_ = 1.0f / std::sqrt(query_pre_attn_scalar);

    if (!is_sliding_) {
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(
            num_attention_heads_, head_dim_, scale_, num_key_value_heads_, layer_idx_,
            kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_);
    } else {
        if (attention_backend_ != ::infinilm::backends::AttentionBackend::STATIC_ATTN) {
            throw std::runtime_error(
                "infinilm::models::gemma3::Gemma3Attention: sliding-window attention requires the STATIC_ATTN backend");
        }
        // Sliding layers bypass StaticAttentionImpl and write the cache
        // directly in the activation dtype; with an int8 KV cache those
        // stores would be silently truncated, so refuse the combination
        // instead of producing garbage attention.
        if (model_config->get_kv_quant_scheme() != infinilm::quantization::KVQuantAlgo::NONE) {
            throw std::runtime_error(
                "infinilm::models::gemma3::Gemma3Attention: KV-cache quantization is not supported for "
                "sliding-window attention layers");
        }
    }

    infinilm::layers::attention::init_kv_cache_quant_params(register_fn, device, kv_cache_k_scale_, kv_cache_v_scale_);
}

std::tuple<infinicore::Tensor, infinicore::Tensor, infinicore::Tensor>
Gemma3Attention::project_and_rotate_(const infinicore::Tensor &positions,
                                     const infinicore::Tensor &hidden_states) const {
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    // QK-norm on per-head vectors (before RoPE), mirroring the HF reference.
    q = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
    k = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));

    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    auto pos_shape = positions->shape();
    infinicore::Tensor pos_ids_for_rope = positions;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = positions->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = positions->contiguous();
    } else {
        throw std::runtime_error("infinilm::models::gemma3::Gemma3Attention: Unexpected position_ids shape");
    }

    rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    return {q_reshaped, k_reshaped, v_reshaped};
}

infinicore::Tensor Gemma3Attention::forward(const infinicore::Tensor &positions,
                                            const infinicore::Tensor &hidden_states) const {
    auto [q, k, v] = project_and_rotate_(positions, hidden_states);
    if (!is_sliding_) {
        return forward_full_(q, k, v);
    }
    return forward_sliding_(q, k, v);
}

infinicore::Tensor Gemma3Attention::forward_full_(infinicore::Tensor &q_reshaped,
                                                  infinicore::Tensor &k_reshaped,
                                                  infinicore::Tensor &v_reshaped) const {
    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped);
    return o_proj_->forward(attn_output);
}

infinicore::Tensor Gemma3Attention::forward_sliding_(infinicore::Tensor &q_reshaped,
                                                     infinicore::Tensor &k_reshaped,
                                                     infinicore::Tensor &v_reshaped) const {
    // q/k/v: [bs, seq, heads, head_dim]
    auto shape = q_reshaped->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];
    size_t ngroup = num_attention_heads_ / num_key_value_heads_;

    auto &forward_context = infinilm::global_state::get_forward_context();
    auto &kv_cache = forward_context.kv_cache_vec[layer_idx_];
    auto &attn_metadata = forward_context.attn_metadata;
    size_t past_len = read_first_len_(attn_metadata.past_sequence_lengths.value());
    size_t total_len = past_len + seq_len;

    // KV cache update, mirroring StaticAttentionImpl::do_kv_cache_update.
    auto k_perm = k_reshaped->permute({0, 2, 1, 3}); // [bs, nkv, seq, head_dim]
    auto v_perm = v_reshaped->permute({0, 2, 1, 3});
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0); // [bs, nkv, max_len, head_dim]
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    // Same capacity guard as StaticAttentionImpl::do_kv_cache_update.
    ASSERT(past_len + seq_len <= k_cache_layer->size(2));
    k_cache_layer->narrow({{2, past_len, seq_len}})->copy_from(k_perm);
    v_cache_layer->narrow({{2, past_len, seq_len}})->copy_from(v_perm);

    // q_reshaped is [bs, seq, heads, dim] (seq-major); regroup to
    // [bs*nkv, ng*seq, dim] head-major so GQA groups align with the cache.
    auto Q = q_reshaped->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});

    infinicore::Tensor attn_output;
    if (seq_len == 1) {
        // Decode: only the trailing window is visible, so slice the cache and
        // skip the mask entirely.
        size_t start = total_len > sliding_window_ ? total_len - sliding_window_ : 0;
        size_t window_len = total_len - start;
        auto K = k_cache_layer->narrow({{2, start, window_len}})->view({batch_size * num_key_value_heads_, window_len, head_dim_});
        auto V = v_cache_layer->narrow({{2, start, window_len}})->view({batch_size * num_key_value_heads_, window_len, head_dim_});
        auto K_transposed = K->permute({0, 2, 1});
        auto scores = infinicore::op::matmul(Q, K_transposed, scale_); // [bs*nkv, ng, win]
        auto scores_viewed = scores->view({batch_size * num_attention_heads_, seq_len, window_len});
        infinicore::op::causal_softmax_(scores_viewed, scores_viewed);
        auto out = infinicore::op::matmul(scores, V); // [bs*nkv, ng, head_dim]
        attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                          ->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
    } else {
        // Prefill: trim the key range to the union of visible keys (keys older
        // than past-window+1 are masked for every query), then apply the
        // window mask; causal_softmax_ provides the aligned-causal part.
        size_t k_start = 0; // BISECT: truncation disabled
        size_t k_len = total_len - k_start;
        auto K = k_cache_layer->narrow({{2, k_start, k_len}})->view({batch_size * num_key_value_heads_, k_len, head_dim_});
        auto V = v_cache_layer->narrow({{2, k_start, k_len}})->view({batch_size * num_key_value_heads_, k_len, head_dim_});
        auto K_transposed = K->permute({0, 2, 1});
        auto scores = infinicore::op::matmul(Q, K_transposed, scale_); // [bs*nkv, ng*seq, k_len]

        auto scores_viewed = scores->view({batch_size * num_attention_heads_, seq_len, k_len});
        // causal_softmax_ provides the aligned-causal mask and the softmax; the
        // additive mask only needs to enforce the window constraint
        // (key j is invisible to query i when past+i-j >= sliding_window).
        infinicore::Tensor mask = sliding_mask_(seq_len, past_len, k_start, k_len, scores_viewed->dtype(), scores_viewed->device());
        mask = infinicore::op::broadcast_to(
            mask->view({static_cast<int64_t>(1), static_cast<int64_t>(seq_len), static_cast<int64_t>(k_len)}),
            {static_cast<int64_t>(batch_size * num_attention_heads_),
             static_cast<int64_t>(seq_len),
             static_cast<int64_t>(k_len)});
        auto scores_masked = infinicore::op::add(scores_viewed, mask);
        infinicore::op::causal_softmax_(scores_masked, scores_masked);

        auto scores_grouped = scores_masked->view({batch_size * num_key_value_heads_, ngroup * seq_len, k_len});
        auto out = infinicore::op::matmul(scores_grouped, V); // [bs*nkv, ng*seq, head_dim]
        attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                          ->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
    }
    return o_proj_->forward(attn_output);
}

namespace {
// bf16 is the truncated top half of a float; build the bit pattern directly so
// the mask can be created in the score dtype without a cast op.
uint16_t to_bf16_bits(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}
} // namespace

infinicore::Tensor Gemma3Attention::sliding_mask_(size_t seq, size_t past, size_t k_start,
                                                  size_t k_len, const infinicore::DataType &dtype,
                                                  const infinicore::Device &device) const {
    const size_t total = k_start + k_len;
    // Masks depend only on (seq, total, window), so they are cached per shape
    // to avoid rebuilding the host buffer on every prefill. The cache is keyed
    // by shape and bounded in practice by the distinct prompt lengths seen;
    // clear it if it grows beyond a sane bound.
    if (mask_cache_.size() > 128) {
        mask_cache_.clear();
    }
    size_t key = seq * 1000000000ULL + total;
    auto it = mask_cache_.find(key);
    if (it != mask_cache_.end()) {
        return it->second;
    }

    auto host = infinicore::Tensor::empty({seq, total}, dtype, infinicore::Device::cpu());
    if (dtype == infinicore::DataType::F32) {
        auto *data = reinterpret_cast<float *>(host->data());
        for (size_t i = 0; i < seq; ++i) {
            size_t q_pos = past + i;
            for (size_t j = 0; j < k_len; ++j) {
                size_t key_pos = k_start + j;
                bool visible = (key_pos <= q_pos) && (q_pos - key_pos < sliding_window_);
                data[i * k_len + j] = visible ? 0.0f : kMaskValue;
            }
        }
    } else if (dtype == infinicore::DataType::BF16) {
        auto *data = reinterpret_cast<uint16_t *>(host->data());
        const uint16_t masked = to_bf16_bits(kMaskValue);
        for (size_t i = 0; i < seq; ++i) {
            size_t q_pos = past + i;
            for (size_t j = 0; j < k_len; ++j) {
                size_t key_pos = k_start + j;
                bool visible = (key_pos <= q_pos) && (q_pos - key_pos < sliding_window_);
                data[i * k_len + j] = visible ? 0 : masked;
            }
        }
    } else if (dtype == infinicore::DataType::F16) {
        // -1e9 overflows the fp16 range; use fp16 -inf (0xFC00) instead.
        auto *data = reinterpret_cast<uint16_t *>(host->data());
        constexpr uint16_t kFp16NegInf = 0xFC00;
        for (size_t i = 0; i < seq; ++i) {
            size_t q_pos = past + i;
            for (size_t j = 0; j < k_len; ++j) {
                size_t key_pos = k_start + j;
                bool visible = (key_pos <= q_pos) && (q_pos - key_pos < sliding_window_);
                data[i * k_len + j] = visible ? 0 : kFp16NegInf;
            }
        }
    } else {
        throw std::runtime_error("infinilm::models::gemma3::Gemma3Attention::sliding_mask_: unsupported dtype");
    }
    infinicore::Tensor mask = host->to(device);
    mask_cache_.emplace(key, mask);
    return mask;
}

} // namespace infinilm::models::gemma3
