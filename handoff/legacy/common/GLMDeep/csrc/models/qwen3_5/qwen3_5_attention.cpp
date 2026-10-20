#include "execution/submit.hpp"
#include "../../weights/weights.hpp"
#include "../../layers/normalization/rmsnorm.hpp"
#include "qwen3_5_attention.hpp"
#include "runtime/runtime.hpp"
#include "../../layers/attention/attention.hpp"
#include "../../layers/rotary_embedding/rotary_embedding.hpp"
#include "../../layers/rotary_embedding/rotary_embedding_factory.hpp"
#include "../../utils.hpp"
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <optional>
#include <vector>

namespace infinilm::models::qwen3_5 {

Qwen35Attention::Qwen35Attention(infinilm::runtime::Runtime &runtime,
                                  std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                  size_t layer_idx, const engine::distributed::RankInfo &rank_info,
                                  ::infinilm::backends::AttentionBackend attention_backend) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording()) throw std::logic_error("Attention construction requires eager Runtime");
    if (!model_config || rank_info.device != runtime.device() || rank_info.tp_size <= 0
        || rank_info.tp_rank < 0 || rank_info.tp_rank >= rank_info.tp_size
        || (rank_info.tp_size > 1 && !rank_info.comm))
        throw std::invalid_argument("Invalid explicit Qwen35Attention construction resources");
    if (attention_backend != ::infinilm::backends::AttentionBackend::STATIC_ATTN
        && attention_backend != ::infinilm::backends::AttentionBackend::PAGED_ATTN
        && attention_backend != ::infinilm::backends::AttentionBackend::FLASH_ATTN)
        throw std::invalid_argument("Unsupported explicit Attention backend");
    for (const char *key : {"hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads", "max_position_embeddings"})
        if (model_config->get<int64_t>(key) <= 0) throw std::invalid_argument("Qwen35Attention requires positive dimensions");
    const size_t hidden = model_config->get<size_t>("hidden_size"), dim = model_config->get<size_t>("head_dim");
    const size_t heads = model_config->get<size_t>("num_attention_heads"), kv = model_config->get<size_t>("num_key_value_heads");
    const size_t tp = static_cast<size_t>(rank_info.tp_size);
    if (dim < 2 || heads % kv || heads % tp || (kv % tp && tp % kv))
        throw std::invalid_argument("Qwen35Attention TP/GQA/RoPE geometry is invalid");
    const auto dtype = model_config->get_dtype();
    if (dtype != infinilm::runtime::DataType::kFloat32 && dtype != infinilm::runtime::DataType::kFloat16
        && dtype != infinilm::runtime::DataType::kBFloat16)
        throw std::invalid_argument("Attention RoPE requires floating model dtype");
    const size_t limit = std::numeric_limits<size_t>::max();
    const auto product = [limit](size_t a, size_t b) {
        if (b && a > limit / b) throw std::length_error("Qwen35Attention construction size overflow");
        return a * b;
    };
    const size_t q = product(heads, dim), k = product(product(kv, dim), kv < tp ? tp / kv : 1);
    const size_t q_gate = product(q, 2), kv_pair = product(k, 2);
    if (q_gate > limit - kv_pair) throw std::length_error("Qwen35Attention fused QKV size overflow");
    (void)product(product(q_gate + kv_pair, hidden), infinilm::runtime::dsize(dtype));
    (void)product(product(q, hidden), infinilm::runtime::dsize(dtype));

    // Preflight the existing RoPE and KV metadata before any Parameter allocation.
    const auto &rope_params = model_config->get_config_json()["rope_parameters"];
    const double partial_rotary_factor = rope_params["partial_rotary_factor"].get<double>();
    const double scaled_rotary = static_cast<double>(dim) * partial_rotary_factor;
    if (!std::isfinite(scaled_rotary) || scaled_rotary < 0
        || scaled_rotary >= static_cast<double>(std::numeric_limits<long long>::max()))
        throw std::invalid_argument("Qwen35Attention rotary dimension is invalid");
    size_t rotary = static_cast<size_t>(std::llround(scaled_rotary));
    rotary = std::clamp(rotary, static_cast<size_t>(2), dim);
    if (rotary % 2 != 0) rotary -= 1;
    rotary = std::max(rotary, static_cast<size_t>(2));
    const auto mrope_section = rope_params["mrope_section"].get<std::vector<int>>();
    if (mrope_section.size() != 3)
        throw std::runtime_error("infinilm::models::qwen3_5::Qwen35Attention: mrope_section must have 3 elements");
    size_t section_sum = 0;
    for (int section : mrope_section) {
        if (section <= 0 || static_cast<size_t>(section) > limit - section_sum)
            throw std::invalid_argument("mrope_section must contain 3 positive values");
        section_sum += static_cast<size_t>(section);
    }
    if (section_sum != rotary / 2) throw std::invalid_argument("MRoPE section sum must equal rotary_dim / 2");
    const bool mrope_interleaved = rope_params["mrope_interleaved"].get<bool>();
    const double rope_theta = rope_params["rope_theta"].get<double>();
    const size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
    const double rms_norm_eps = model_config->get<double>("rms_norm_eps");
    const auto kv_scheme = model_config->get_kv_quant_scheme();
    if (kv_scheme != infinilm::quantization::KVQuantAlgo::NONE && kv_scheme != infinilm::quantization::KVQuantAlgo::INT8)
        throw std::invalid_argument("unsupported kv_quant_scheme");
    auto quantization_method = model_config->get_quantization_method();
    auto rope_scaling = infinilm::layers::rotary_embedding::make_scaling_config(model_config);
    construction_owner_ = &runtime;
    device_ = runtime.device();
    layer_idx_ = layer_idx; hidden_size_ = hidden; head_dim_ = dim; rotary_dim_ = rotary;
    num_attention_heads_ = heads / tp; num_key_value_heads_ = kv < tp ? 1 : kv / tp;
    attention_backend_ = attention_backend;
    try {
        auto register_fn = [this](const std::string &n, infinilm::weights::Weight p) { this->bind_weight(n, std::move(p)); };
        qkv_proj_ = std::make_shared<Qwen35FusedQKVLinear>(runtime,
            hidden, dim, heads, kv, "q_proj", "k_proj", "v_proj", register_fn,
            quantization_method, model_config->get_or<bool>("attention_bias", true), dtype, rank_info);
        o_proj_ = std::make_shared<layers::linear::RowParallelLinear>(runtime,
            q, hidden, quantization_method, model_config->get_or<bool>("attention_output_bias", false),
            dtype, rank_info.tp_rank, rank_info.tp_size, rank_info.comm);
        this->bind_child("o_proj", o_proj_);
        mrope_ = infinilm::layers::rotary_embedding::get_rope(runtime, model_config, dim, rotary,
            max_position_embeddings, rope_theta, model_config->get_rope_algo(), dtype, runtime.device(),
            rope_scaling, mrope_section, mrope_interleaved);
        q_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, dim, rms_norm_eps, dtype);
        this->bind_child("q_norm", q_norm_);
        k_norm_ = std::make_shared<infinilm::layers::RMSNorm>(runtime, dim, rms_norm_eps, dtype);
        this->bind_child("k_norm", k_norm_);
        // Existing typed helper publishes into owning members before the backend
        // captures Tensor scale values. Every completed child remains a member.
        infinilm::layers::attention::init_kv_cache_quant_params(runtime, register_fn,
            kv_cache_k_scale_, kv_cache_v_scale_, kv_scheme);
        const float scaling = 1.0f / std::sqrt(static_cast<float>(dim));
        attn_ = std::make_shared<infinilm::layers::attention::AttentionLayer>(num_attention_heads_, dim,
            scaling, num_key_value_heads_, layer_idx_, kv_cache_k_scale_, kv_cache_v_scale_, attention_backend_, kv_scheme);
    } catch (...) {
        const auto failure = std::current_exception();
        // Constructor body catch, not a function-try handler:
        // all completed members and Parameter owners outlive this owner drain.
        try { runtime.syncStream(); } catch (...) { std::terminate(); }
        std::rethrow_exception(failure);
    }
}

infinilm::runtime::Tensor Qwen35Attention::forward(const infinilm::runtime::Tensor &positions,
                                            const infinilm::runtime::Tensor &hidden_states,
                                            infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    runtime.assertOwnerThread();
    if (&runtime != construction_owner_)
        throw std::invalid_argument("Qwen35Attention forward Runtime differs from construction owner");
    if (!hidden_states || !positions || runtime.device() != device_
        || hidden_states->device() != runtime.device() || positions->device() != runtime.device())
        throw std::invalid_argument("Qwen35Attention Runtime/input/positions devices differ");
    if (hidden_states->shape().size() != 3 || hidden_states->size(2) != hidden_size_)
        throw std::invalid_argument("Qwen35Attention expects [batch, sequence, hidden_size]");
    const auto pos_rank = positions->shape().size();
    if (pos_rank != 1 && pos_rank != 2) {
        if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_)
            throw std::runtime_error("infinilm::models::qwen3_5::Qwen35Attention: Unexpected position_ids shape");
        throw std::runtime_error("Unexpected position_ids shape");
    }
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN != attention_backend_ && hidden_states->size(0) != 1)
        throw std::invalid_argument("Qwen35Attention paged forward requires batch size 1");
    if (::infinilm::backends::AttentionBackend::STATIC_ATTN == attention_backend_) {
        return forward_static_(positions, hidden_states, forward_context);
    }
    return forward_paged_(positions, hidden_states, forward_context);
}

infinilm::runtime::Tensor Qwen35Attention::forward_static_(const infinilm::runtime::Tensor &position_ids,
                                                    const infinilm::runtime::Tensor &hidden_states,
                                                    infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    auto [q, gate, k, v] = qkv_proj_->forward_split(runtime, hidden_states_mutable);

    auto q_reshaped = q_norm_->forward(runtime, q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
    auto k_reshaped = k_norm_->forward(runtime, k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));

    std::tie(q_reshaped, k_reshaped) = mrope_->forward(runtime, q_reshaped, k_reshaped, position_ids);

    q_reshaped = q_reshaped->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    k_reshaped = k_reshaped->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);
    infinilm::layers::attention::detail::apply_output_gate(runtime, attn_output, gate, true);
    return o_proj_->forward(runtime, attn_output);
}

infinilm::runtime::Tensor Qwen35Attention::forward_paged_(const infinilm::runtime::Tensor &position_ids,
                                                   const infinilm::runtime::Tensor &hidden_states,
                                                   infinilm::global_state::ForwardContext &forward_context) const {
    auto &runtime = forward_context.require_runtime();
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t seq_len = shape[1];

    auto [q, gate, k, v] = qkv_proj_->forward_split(runtime, hidden_states_mutable);

    auto q_reshaped = q->view({seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({seq_len, num_key_value_heads_, head_dim_});
    q_reshaped = q_norm_->forward(runtime, q_reshaped);
    k_reshaped = k_norm_->forward(runtime, k_reshaped);

    std::tie(q_reshaped, k_reshaped) = mrope_->forward(runtime, q_reshaped, k_reshaped, position_ids);

    auto attn_output = attn_->forward(q_reshaped, k_reshaped, v_reshaped, forward_context);
    infinilm::layers::attention::detail::apply_output_gate(runtime, attn_output, gate, true);
    return o_proj_->forward(runtime, attn_output);
}
} // namespace infinilm::models::qwen3_5
