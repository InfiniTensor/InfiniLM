#include "minicpm_sala_attention.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/infllmv2_attention.hpp"
#include "infinicore/ops/simple_gla_attention.hpp"
#include "infinicore/ops/simple_gla_decode_step.hpp"
#include "infinicore/ops/simple_gla_prefill.hpp"
#include "infinicore/ops/simple_gla_recurrent_state_append.hpp"
#include "infinicore/context/context.hpp"
#include "../../global_state/global_state.hpp"
#include "../debug_utils/tensor_utils.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace infinilm::models::minicpm_sala {

namespace {

// Per-layer KV tensor layout from `StaticKVCache::create_layer_kv_cache`: [2, B, n_kv, max_len, D].
void minicpm_sala_update_layer_kv_tensor(infinicore::Tensor &kv_bundle,
                                         const infinicore::Tensor &k_permuted,
                                         const infinicore::Tensor &v_permuted,
                                         const infinicore::Tensor &past_sequence_lengths) {
    auto k_cache_layer = kv_bundle->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_bundle->narrow({{0, 1, 1}})->squeeze(0);

#ifdef ENABLE_KV_CACHING
    infinicore::op::kv_caching_(
        k_cache_layer,
        v_cache_layer,
        k_permuted,
        v_permuted,
        past_sequence_lengths);
#else
    const size_t cache_pos = static_cast<size_t>(
        reinterpret_cast<int32_t *>(past_sequence_lengths->to(infinicore::Device::cpu())->data())[0]);
    const size_t update_len = k_permuted->size(2);
    const size_t result_len = cache_pos + update_len;
    if (result_len > k_cache_layer->size(2)) {
        throw std::runtime_error("MiniCPMSALAAttention(KV update): KV cache length exceeded");
    }
    k_cache_layer->narrow({{2, cache_pos, update_len}})->copy_from(k_permuted);
    v_cache_layer->narrow({{2, cache_pos, update_len}})->copy_from(v_permuted);
#endif
}

// Same as HF MiniCPM-SALA _build_slope_tensor (used for Simple GLA decay).
std::vector<float> build_slope_tensor(size_t n) {
    auto get_slopes_power_of_2 = [](size_t n) -> std::vector<float> {
        double log2n = std::log2(static_cast<double>(n));
        double start = std::pow(2.0, -(std::pow(2.0, -(log2n - 3))));
        double ratio = start;
        std::vector<float> out;
        out.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            out.push_back(static_cast<float>(start * std::pow(ratio, static_cast<double>(i))));
        }
        return out;
    };
    if (n == 0) return {};
    double log2n = std::log2(static_cast<double>(n));
    if (std::abs(log2n - std::floor(log2n)) < 1e-9) {
        return get_slopes_power_of_2(n);
    }
    size_t closest = static_cast<size_t>(std::pow(2.0, std::floor(log2n)));
    auto first = get_slopes_power_of_2(closest);
    auto rest = build_slope_tensor(2 * closest);
    for (size_t i = 0; i < n - closest; ++i) {
        first.push_back(rest[i * 2]);
    }
    return first;
}

} // namespace

namespace {
void ensure_gla_state_allocated(infinicore::Tensor &state,
                               const infinicore::Device &device,
                               size_t batch_size,
                               size_t n_h,
                               size_t head_dim) {
    const std::vector<size_t> want = {batch_size, n_h, head_dim, head_dim};
    if (!state || state->shape() != want || state->dtype() != infinicore::DataType::F32 || state->device() != device) {
        state = infinicore::Tensor::zeros(want, infinicore::DataType::F32, device);
    }
}
} // namespace

MiniCPMSALALightningAttention::MiniCPMSALALightningAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                             const infinicore::Device &device,
                                                             size_t layer_idx)
    : layer_idx_(layer_idx) {
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");

    num_attention_heads_ = model_config->get_or<size_t>("lightning_nh", model_config->get<size_t>("num_attention_heads"));
    num_key_value_heads_ = model_config->get_or<size_t>("lightning_nkv", model_config->get<size_t>("num_key_value_heads"));
    head_dim_ = model_config->get_or<size_t>("lightning_head_dim", model_config->get<size_t>("head_dim"));
    scaling_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    use_rope_ = model_config->get_or<bool>("lightning_use_rope", true);
    rotary_emb_ = infinilm::layers::rotary_embedding::get_rope(model_config, device);

    use_qk_norm_ = model_config->get_or<bool>("qk_norm", true);
    use_output_gate_ = model_config->get_or<bool>("use_output_gate", true);

    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size, num_attention_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size, num_key_value_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size, num_key_value_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size, false, dtype, device);

    if (use_qk_norm_) {
        INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, model_config->get<double>("rms_norm_eps"), dtype, device);
        INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, model_config->get<double>("rms_norm_eps"), dtype, device);
    }
    use_output_norm_ = true;
    INFINICORE_NN_MODULE_INIT(o_norm, hidden_size, model_config->get<double>("rms_norm_eps"), dtype, device);
    INFINICORE_NN_MODULE_INIT(z_proj, hidden_size, hidden_size, false, dtype, device);

    std::vector<float> slopes = build_slope_tensor(num_attention_heads_);
    auto g_cpu = infinicore::Tensor::empty(
        {num_attention_heads_}, infinicore::DataType::F32, infinicore::Device::cpu());
    float *ptr = reinterpret_cast<float *>(g_cpu->data());
    for (size_t h = 0; h < num_attention_heads_; ++h)
        ptr[h] = -slopes[h];
    g_gamma_ = g_cpu->to(device);
}

void MiniCPMSALALightningAttention::reset_state() {
    gla_state_valid_ = false;
    gla_state_cached_len_ = 0;
    gla_state_ = {};
}

infinicore::Tensor MiniCPMSALALightningAttention::forward(const infinicore::Tensor &position_ids,
                                                         const infinicore::Tensor &hidden_states) const {
    const auto &attn_meta = infinilm::global_state::get_forward_context().attn_metadata;
    auto past_sequence_lengths = attn_meta.past_sequence_lengths;
    auto total_sequence_lengths = attn_meta.total_sequence_lengths;
    auto cu_seqlens = attn_meta.cu_seqlens;
    // input_offsets/block_tables/slot_mapping are not used in this dense/per-layer-kv implementation yet.
    (void)cu_seqlens;
    // Input: [B, S, H]
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto hs_mut = hidden_states;
    auto q = q_proj_->forward(hs_mut);
    auto k = k_proj_->forward(hs_mut);
    auto v = v_proj_->forward(hs_mut);
    // View requires contiguous layout; only call contiguous when needed (proj output often already contiguous).
    auto q_reshaped = q->contiguous()->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->contiguous()->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->contiguous()->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    if (use_qk_norm_) {
        // RMSNorm op only supports 2D/3D; normalize over head_dim with a 3D view.
        auto q3 = q_reshaped->view({batch_size * seq_len, num_attention_heads_, head_dim_});
        auto k3 = k_reshaped->view({batch_size * seq_len, num_key_value_heads_, head_dim_});
        q3 = q_norm_->forward(q3);
        k3 = k_norm_->forward(k3);
        q_reshaped = q3->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        k_reshaped = k3->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    }

    // RoPE only for lightning layers (HyPE)
    if (use_rope_) {
        if (!rotary_emb_) {
            throw std::runtime_error("MiniCPMSALALightningAttention: rotary_emb is not set but use_rope=true");
        }
        // position_ids can be [B,S] or [S]; follow LlamaAttention behavior.
        auto pos_shape = position_ids->shape();
        infinicore::Tensor pos_ids_for_rope = position_ids;
        if (pos_shape.size() == 2) {
            auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
            pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
        } else if (pos_shape.size() == 1) {
            pos_ids_for_rope = position_ids->contiguous();
        } else {
            throw std::runtime_error("MiniCPMSALALightningAttention: Unexpected position_ids shape");
        }

        rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    }

    // Compute dense attention (GQA): reshape as LlamaAttention does
    size_t total_seq_len = seq_len;
    size_t cache_pos = 0;
    const bool has_cache_meta = past_sequence_lengths.has_value() && total_sequence_lengths.has_value();
    if (has_cache_meta) {
        auto past_cpu = past_sequence_lengths.value()->to(infinicore::Device::cpu());
        cache_pos = reinterpret_cast<int32_t *>(past_cpu->data())[0];
        // `total_sequence_lengths` may be input length (e.g. 1 on decode); KV length is cache_pos + seq_len.
        total_seq_len = cache_pos + seq_len;
    } else if (total_sequence_lengths.has_value()) {
        total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
    }

    // Cache expects [B, n_kv, S, D]. Keep this as a strided view and let the caching op handle strides
    // to avoid a full rearrange (permute->contiguous) copy on long-context prefill.
    // Correctness: kv_caching_ / StaticKVCache::update is sensitive to input stride/layout.
    // Restore contiguous to match HF logits exactly before re-applying any strided optimizations.
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3})->contiguous(); // [B, n_kv, S, D]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3})->contiguous(); // [B, n_kv, S, D]

    // Per-layer KV tensors in `global_state::get_forward_context().kv_cache_vec` (same pattern as
    // `InfinilmModel::reset_cache` / `StaticAttentionImpl`).
    infinicore::Tensor k_total = k_permuted;
    infinicore::Tensor v_total = v_permuted;
    bool use_forward_kv = false;
    if (has_cache_meta) {
        auto &kv_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
        if (layer_idx_ >= kv_vec.size()) {
            throw std::runtime_error(
                "MiniCPMSALALightningAttention: forward_context.kv_cache_vec is unset or too small (call reset_cache / align layer count)");
        }
        use_forward_kv = true;
        minicpm_sala_update_layer_kv_tensor(
            kv_vec[layer_idx_],
            k_permuted,
            v_permuted,
            past_sequence_lengths.value());
        auto k_cache_layer = kv_vec[layer_idx_]->narrow({{0, 0, 1}})->squeeze(0);
        auto v_cache_layer = kv_vec[layer_idx_]->narrow({{0, 1, 1}})->squeeze(0);
        k_total = k_cache_layer;
        v_total = v_cache_layer;
    } else {
        total_seq_len = seq_len;
    }

    // Slice to total_seq_len (decode-only / cont-batch)
    if (total_seq_len > k_total->shape()[2]) {
        throw std::runtime_error("MiniCPMSALALightningAttention: total_seq_len exceeds available KV length (cache not correctly updated)");
    }
    k_total = k_total->narrow({{2, 0, total_seq_len}});
    v_total = v_total->narrow({{2, 0, total_seq_len}});

    infinicore::Tensor attn_output;
    {
        // Lightning-attn only: Simple GLA (HF-aligned).
        // simple_gla_attention(q,k,v,g_gamma,scale) expects [B, T, H, D]; g_gamma [H].
        const size_t n_h = num_attention_heads_;
        const size_t n_kv = num_key_value_heads_;
        infinicore::Tensor k_use = k_total;
        infinicore::Tensor v_use = v_total;
        if (n_kv < n_h) {
            // Repeat KV heads to match n_h (same as HF repeat_kv / repeat_interleave).
            // Use as_strided view then contiguous() so one copy instead of n_h narrow/copy_from calls.
            const size_t ngroup = n_h / n_kv;
            const std::vector<ptrdiff_t> repeat_strides = {
                static_cast<ptrdiff_t>(n_kv * total_seq_len * head_dim_),
                static_cast<ptrdiff_t>(total_seq_len * head_dim_),
                0,
                static_cast<ptrdiff_t>(head_dim_),
                1,
            };
            k_use = k_total->as_strided(
                         {batch_size, n_kv, ngroup, total_seq_len, head_dim_}, repeat_strides)
                         ->contiguous()
                         ->view({batch_size, n_h, total_seq_len, head_dim_});
            v_use = v_total->as_strided(
                         {batch_size, n_kv, ngroup, total_seq_len, head_dim_}, repeat_strides)
                         ->contiguous()
                         ->view({batch_size, n_h, total_seq_len, head_dim_});
        }
        // GLA expects [B, S, H, D]. `q_reshaped` is already [B, S, H, D], so avoid permute+contiguous.
        auto q_bthd = q_reshaped;                                 // [B, S_q, H, D]
        // Correctness: restore contiguous layout for K/V before `simple_gla_attention`.
        auto k_bthd = k_use->permute({0, 2, 1, 3})->contiguous(); // [B, S_kv, H, D]
        auto v_bthd = v_use->permute({0, 2, 1, 3})->contiguous(); // [B, S_kv, H, D]

        // Lightning fast decode: maintain recurrent state locally (do NOT depend on StaticKVCache extensions).
        // We rebuild state on-demand if it is out-of-sync with cache_pos.
        const bool is_decode = has_cache_meta && use_forward_kv && (seq_len == 1) && (total_seq_len > 1);
        if (is_decode) {
            ensure_gla_state_allocated(gla_state_, q_bthd->device(), batch_size, n_h, head_dim_);

            // Ensure `state` corresponds to exactly `cache_pos` cached tokens (excluding current token).
            if (!gla_state_valid_ || gla_state_cached_len_ != cache_pos) {
                // Rebuild from available KV. This is O(T) once after reset / mismatch.
                infinicore::op::zeros_(gla_state_);
                if (cache_pos > 0) {
                    auto k_prev = k_bthd->narrow({{1, 0, cache_pos}});
                    auto v_prev = v_bthd->narrow({{1, 0, cache_pos}});
                    infinicore::op::simple_gla_recurrent_state_append_segment(gla_state_, k_prev, v_prev, g_gamma_);
                }
                gla_state_cached_len_ = cache_pos;
                gla_state_valid_ = true;
            }

            // Decode-step uses only the newest KV at position (total_seq_len - 1).
            auto q_new = q_bthd; // [B,1,H,D]
            auto k_new = k_bthd->narrow({{1, total_seq_len - 1, 1}});
            auto v_new = v_bthd->narrow({{1, total_seq_len - 1, 1}});
            auto out_b1hd = infinicore::op::simple_gla_decode_step(q_new, k_new, v_new, gla_state_, g_gamma_, scaling_);
            gla_state_cached_len_ = cache_pos + 1;
            attn_output = out_b1hd->view({batch_size, seq_len, n_h * head_dim_});
            // Fall through to output norm/gate + o_proj below (do not run full-sequence GLA again).
        } else {
            // Prefill / non-decode batching: non-recurrent kernels, then update local recurrent state.
            infinicore::Tensor q_full;
            if (seq_len == total_seq_len) {
                q_full = q_bthd;
            } else {
                // q shorter than KV: pad q to [B, total_seq_len, H, D].
                q_full = infinicore::Tensor::zeros(
                    {batch_size, total_seq_len, n_h, head_dim_}, q_bthd->dtype(), q_bthd->device());
                auto q_slot = q_full->narrow({{1, total_seq_len - seq_len, seq_len}});
                q_slot->copy_from(q_bthd);
            }

            infinicore::Tensor gla_out;
            // Fused prefill: naive kernel for head_dim<=64; chunked/tiled kernel for head_dim>64 (e.g. 128).
            bool use_fused_prefill = (batch_size == 1) && (seq_len == total_seq_len);
            if (use_fused_prefill) {
                gla_out = infinicore::op::simple_gla_prefill(q_full, k_bthd, v_bthd, g_gamma_, scaling_);
            } else {
                gla_out = infinicore::op::simple_gla_attention(q_full, k_bthd, v_bthd, g_gamma_, scaling_);
            }

            // Keep local recurrent state in sync for subsequent decode steps.
            ensure_gla_state_allocated(gla_state_, q_bthd->device(), batch_size, n_h, head_dim_);
            if (cache_pos == 0) {
                infinicore::op::zeros_(gla_state_);
                gla_state_cached_len_ = 0;
                gla_state_valid_ = true;
            }
            // Append the segment we just wrote: [cache_pos, cache_pos + seq_len)
            if (gla_state_valid_ && gla_state_cached_len_ == cache_pos) {
                auto k_seg = k_bthd->narrow({{1, cache_pos, seq_len}});
                auto v_seg = v_bthd->narrow({{1, cache_pos, seq_len}});
                infinicore::op::simple_gla_recurrent_state_append_segment(gla_state_, k_seg, v_seg, g_gamma_);
                gla_state_cached_len_ = cache_pos + seq_len;
            } else {
                // Out-of-sync; force rebuild next time we need recurrent decode.
                gla_state_valid_ = false;
            }

            infinicore::Tensor out_slice = gla_out->narrow({{1, total_seq_len - seq_len, seq_len}});
            attn_output = out_slice->view({batch_size, seq_len, n_h * head_dim_});
        }
    }

    // Lightning output gate/norm
    if (use_output_gate_) {
        auto z_in = hidden_states;
        auto z = z_proj_->forward(z_in);
        infinicore::op::sigmoid_(z, z);
        if (use_output_norm_ && o_norm_) {
            attn_output = o_norm_->forward(attn_output);
        }
        attn_output = infinicore::op::mul(attn_output, z);
    } else if (use_output_norm_ && o_norm_) {
        attn_output = o_norm_->forward(attn_output);
    }

    auto attn_out_mut = attn_output;
    auto out = o_proj_->forward(attn_out_mut);

    return out;
}

MiniCPMSALAMinicpm4Attention::MiniCPMSALAMinicpm4Attention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                           const infinicore::Device &device,
                                                           size_t layer_idx)
    : layer_idx_(layer_idx) {
    (void)device;
    const auto dtype = model_config->get_dtype();
    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    num_attention_heads_ = model_config->get<size_t>("num_attention_heads");
    num_key_value_heads_ = model_config->get<size_t>("num_key_value_heads");
    head_dim_ = model_config->get<size_t>("head_dim");
    scaling_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    int sparse_window_size = model_config->get_or<int>("sparse_window_size", -1);
    if (sparse_window_size <= 0) {
        auto sparse_cfg = model_config->get_or<nlohmann::json>("sparse_config", nlohmann::json{});
        if (!sparse_cfg.is_null() && sparse_cfg.contains("window_size")) {
            sparse_window_size = sparse_cfg["window_size"].get<int>();
        } else {
            sparse_window_size = model_config->get_or<int>("window_size", -1);
        }
    }
    if (sparse_window_size > 0) {
        infllmv2_window_left_ = sparse_window_size;
        use_local_window_ = true;
    }

    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size, num_attention_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size, num_key_value_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size, num_key_value_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(o_gate, hidden_size, hidden_size, false, dtype, device);
}

void MiniCPMSALAMinicpm4Attention::reset_state() {
    // no local recurrent state
}

infinicore::Tensor MiniCPMSALAMinicpm4Attention::forward(const infinicore::Tensor &position_ids,
                                                        const infinicore::Tensor &hidden_states) const {
    (void)position_ids;
    const auto &attn_meta = infinilm::global_state::get_forward_context().attn_metadata;
    auto past_sequence_lengths = attn_meta.past_sequence_lengths;
    auto total_sequence_lengths = attn_meta.total_sequence_lengths;

    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    auto hs_mut = hidden_states;
    auto q = q_proj_->forward(hs_mut);
    auto k = k_proj_->forward(hs_mut);
    auto v = v_proj_->forward(hs_mut);
    auto q_reshaped = q->contiguous()->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->contiguous()->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->contiguous()->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // KV update via per-layer kv_cache_vec when metadata present
    size_t total_seq_len = seq_len;
    size_t cache_pos = 0;
    const bool has_cache_meta = past_sequence_lengths.has_value() && total_sequence_lengths.has_value();
    if (has_cache_meta) {
        auto past_cpu = past_sequence_lengths.value()->to(infinicore::Device::cpu());
        cache_pos = reinterpret_cast<int32_t *>(past_cpu->data())[0];
        total_seq_len = cache_pos + seq_len;
    }
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3})->contiguous();
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3})->contiguous();

    infinicore::Tensor k_total = k_permuted;
    infinicore::Tensor v_total = v_permuted;
    bool use_forward_kv = false;
    if (has_cache_meta) {
        auto &kv_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
        if (layer_idx_ >= kv_vec.size()) {
            throw std::runtime_error(
                "MiniCPMSALAMinicpm4Attention: forward_context.kv_cache_vec is unset or too small");
        }
        use_forward_kv = true;
        minicpm_sala_update_layer_kv_tensor(
            kv_vec[layer_idx_],
            k_permuted,
            v_permuted,
            past_sequence_lengths.value());
        auto k_cache_layer = kv_vec[layer_idx_]->narrow({{0, 0, 1}})->squeeze(0);
        auto v_cache_layer = kv_vec[layer_idx_]->narrow({{0, 1, 1}})->squeeze(0);
        k_total = k_cache_layer;
        v_total = v_cache_layer;
    } else {
        total_seq_len = seq_len;
    }

    if (total_seq_len > k_total->shape()[2]) {
        throw std::runtime_error("MiniCPMSALAMinicpm4Attention: total_seq_len exceeds available KV length");
    }
    k_total = k_total->narrow({{2, 0, total_seq_len}});
    v_total = v_total->narrow({{2, 0, total_seq_len}});

    try {
        if (!total_sequence_lengths.has_value()) {
            throw std::runtime_error("MiniCPMSALAMinicpm4Attention: total_sequence_lengths is required for InfLLM-v2 path");
        }
        const auto cache_lens = total_sequence_lengths.value();
        const bool force_varlen_decode = [&]() {
            const char *env = std::getenv("INFINI_MINICPM4_DECODE_VARLEN");
            return env && env[0] != '\0' && env[0] != '0';
        }();

        infinicore::Tensor attn_output;
        if (seq_len == total_seq_len || (force_varlen_decode && batch_size == 1)) {
            if (batch_size != 1) {
                throw std::runtime_error("MiniCPMSALAMinicpm4Attention: varlen path requires batch_size=1");
            }
            auto q_bshd = q_reshaped->contiguous();
            auto k_btkd = k_total->permute({0, 2, 1, 3})->contiguous();
            auto v_btkd = v_total->permute({0, 2, 1, 3})->contiguous();
            auto q_var = q_bshd->view({static_cast<ptrdiff_t>(seq_len), static_cast<ptrdiff_t>(num_attention_heads_), static_cast<ptrdiff_t>(head_dim_)});
            auto k_var = k_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});
            auto v_var = v_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});

            auto cuq_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
            reinterpret_cast<int32_t *>(cuq_cpu->data())[0] = 0;
            reinterpret_cast<int32_t *>(cuq_cpu->data())[1] = static_cast<int32_t>(seq_len);
            infinicore::Tensor cu_q = cuq_cpu->to(q_var->device());
            auto cuk_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
            reinterpret_cast<int32_t *>(cuk_cpu->data())[0] = 0;
            reinterpret_cast<int32_t *>(cuk_cpu->data())[1] = static_cast<int32_t>(total_seq_len);
            infinicore::Tensor cu_k = cuk_cpu->to(q_var->device());

            const bool infllmv2_causal = !use_local_window_;
            const int window_left = use_local_window_ ? infllmv2_window_left_ : -1;
            const int window_right = use_local_window_ ? 0 : -1;

            auto out_var = infinicore::op::infllmv2_varlen(
                q_var, k_var, v_var,
                cu_q, cu_k,
                static_cast<int>(seq_len),
                static_cast<int>(total_seq_len),
                scaling_,
                /*causal=*/infllmv2_causal,
                /*window_size_left=*/window_left,
                /*window_size_right=*/window_right);
            attn_output = out_var->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        } else if (use_forward_kv) {
            if (batch_size != 1) {
                throw std::runtime_error("MiniCPMSALAMinicpm4Attention: kvcache decode requires batch_size=1");
            }
            auto q_bshd = q_reshaped->contiguous();
            auto k_bthd = k_total->permute({0, 2, 1, 3})->contiguous();
            auto v_bthd = v_total->permute({0, 2, 1, 3})->contiguous();

            const bool infllmv2_causal = !use_local_window_;
            const int window_left = use_local_window_ ? infllmv2_window_left_ : -1;
            const int window_right = use_local_window_ ? 0 : -1;

            auto out_bshd = infinicore::op::infllmv2_kvcache(
                q_bshd,
                k_bthd,
                v_bthd,
                cache_lens,
                scaling_,
                /*causal=*/infllmv2_causal,
                /*window_size_left=*/window_left,
                /*window_size_right=*/window_right);
            attn_output = out_bshd->contiguous()->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
        } else {
            throw std::runtime_error("MiniCPMSALAMinicpm4Attention: decode requires KV cache");
        }

        // Sparse gate + o_proj
        auto gate = o_gate_->forward(hs_mut);
        infinicore::op::sigmoid_(gate, gate);
        attn_output = infinicore::op::mul(attn_output, gate);
        auto out = o_proj_->forward(attn_output);
        return out;
    } catch (const std::exception &e) {
        throw std::runtime_error(
            std::string("MiniCPMSALAMinicpm4Attention: InfLLM-v2 attention failed. ")
            + "Original error: " + e.what());
    }
}

} // namespace infinilm::models::minicpm_sala
