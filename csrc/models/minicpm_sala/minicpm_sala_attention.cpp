#include "minicpm_sala_attention.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/infllmv2_attention.hpp"
#include "infinicore/ops/simple_gla_attention.hpp"
#include "infinicore/ops/simple_gla_prefill.hpp"
#include "infinicore/context/context.hpp"
#include "../debug_utils/tensor_utils.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace infinilm::models::minicpm_sala {

namespace {
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

MiniCPMSALAAttention::MiniCPMSALAAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                           const infinicore::Device &device,
                                           size_t layer_idx,
                                           const std::string &mixer_type,
                                           engine::distributed::RankInfo rank_info,
                                           backends::AttentionBackend attention_backend)
    : model_config_(std::move(model_config)),
      rank_info_(rank_info),
      layer_idx_(layer_idx),
      attention_backend_(attention_backend) {

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config_->get_dtype();
    hidden_size_ = model_config_->get<size_t>("hidden_size");
    if (mixer_type == "minicpm4") {
        is_sparse_layer_ = true;
        num_attention_heads_ = model_config_->get<size_t>("num_attention_heads");
        num_key_value_heads_ = model_config_->get<size_t>("num_key_value_heads");
        head_dim_ = model_config_->get<size_t>("head_dim");

        // InfLLM-v2 local-window masking (causal-local semantics) for minicpm4.
        // Prefer `sparse_window_size`, but fall back to `window_size` if needed.
        int sparse_window_size = model_config_->get_or<int>("sparse_window_size", -1);
        if (sparse_window_size <= 0) {
            // Some HF configs store this under `sparse_config.window_size`.
            auto sparse_cfg = model_config_->get_or<nlohmann::json>("sparse_config", nlohmann::json{});
            if (!sparse_cfg.is_null() && sparse_cfg.contains("window_size")) {
                sparse_window_size = sparse_cfg["window_size"].get<int>();
            } else {
                sparse_window_size = model_config_->get_or<int>("window_size", -1);
            }
        }
        if (sparse_window_size > 0) {
            infllmv2_window_left_ = sparse_window_size;
            infllmv2_window_right_ = 0;
            use_local_window_ = true;
        }
    } else {
        // Lightning layers have their own head config.
        num_attention_heads_ = model_config_->get_or<size_t>("lightning_nh", model_config_->get<size_t>("num_attention_heads"));
        num_key_value_heads_ = model_config_->get_or<size_t>("lightning_nkv", model_config_->get<size_t>("num_key_value_heads"));
        head_dim_ = model_config_->get_or<size_t>("lightning_head_dim", model_config_->get<size_t>("head_dim"));
    }
    scaling_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

    // StaticKVCache is allocated as a compact slab per cache type:
    //  - minicpm4-cache stores only layers where mixer_types[i] == "minicpm4"
    //  - lightning-cache stores only layers where mixer_types[i] != "minicpm4"
    //
    // Compute this attention instance's local cache index (0-based) from its
    // absolute layer_idx_.
    {
        bool this_is_minicpm4_cache = (mixer_type == "minicpm4");
        std::vector<std::string> mixer_types;
        try {
            mixer_types = model_config_->get<std::vector<std::string>>("mixer_types");
        } catch (...) {
            mixer_types.assign(model_config_->get<size_t>("num_hidden_layers"), "minicpm4");
        }
        // Be defensive if mixer_types size mismatches.
        if (mixer_types.size() != model_config_->get<size_t>("num_hidden_layers")) {
            mixer_types.resize(model_config_->get<size_t>("num_hidden_layers"), "minicpm4");
        }
        size_t count = 0;
        for (size_t i = 0; i <= layer_idx_ && i < mixer_types.size(); ++i) {
            const bool is_minicpm4_layer = (mixer_types[i] == "minicpm4");
            if (is_minicpm4_layer == this_is_minicpm4_cache) {
                ++count;
            }
        }
        // layer_idx_ is always a valid layer, so count should be >= 1.
        cache_layer_idx_ = count > 0 ? (count - 1) : 0;
    }

    // HyPE: RoPE in lightning layers, NoPE in sparse (minicpm4) layers.
    // We treat all non-minicpm4 as "linear" (lightning-attn) for M1 dense fallback.
    use_rope_ = (mixer_type != "minicpm4") && model_config_->get_or<bool>("lightning_use_rope", true);

    // MiniCPM-SALA uses QK-norm and output gates by default.
    use_qk_norm_ = model_config_->get_or<bool>("qk_norm", true) && (mixer_type != "minicpm4");
    use_output_gate_ = model_config_->get_or<bool>("use_output_gate", true);

    // Projections
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, num_attention_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, num_key_value_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, num_key_value_heads_ * head_dim_, false, dtype, device);
    INFINICORE_NN_MODULE_INIT(o_proj, num_attention_heads_ * head_dim_, hidden_size_, false, dtype, device);

    if (mixer_type == "minicpm4") {
        // Sparse layers use o_gate (sigmoid gate on attention output)
        INFINICORE_NN_MODULE_INIT(o_gate, hidden_size_, hidden_size_, false, dtype, device);
    } else {
        // Lightning layers use q/k norm + output norm and z-projection gate
        if (use_qk_norm_) {
            INFINICORE_NN_MODULE_INIT(q_norm, head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
            INFINICORE_NN_MODULE_INIT(k_norm, head_dim_, model_config_->get<double>("rms_norm_eps"), dtype, device);
        }
        use_output_norm_ = true;
        // Checkpoint uses o_norm over hidden_size (shape [hidden_size]).
        INFINICORE_NN_MODULE_INIT(o_norm, hidden_size_, model_config_->get<double>("rms_norm_eps"), dtype, device);
        INFINICORE_NN_MODULE_INIT(z_proj, hidden_size_, hidden_size_, false, dtype, device);
    }
    // Simple GLA decay for lightning path: g_gamma = _build_slope_tensor * -1.
    std::vector<float> slopes = build_slope_tensor(num_attention_heads_);
    auto g_cpu = infinicore::Tensor::empty(
        {num_attention_heads_}, infinicore::DataType::F32, infinicore::Device::cpu());
    float *ptr = reinterpret_cast<float *>(g_cpu->data());
    for (size_t h = 0; h < num_attention_heads_; ++h)
        ptr[h] = -slopes[h];
    g_gamma_ = g_cpu->to(device);
}

void MiniCPMSALAAttention::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    rotary_emb_ = rotary_emb;
}

void MiniCPMSALAAttention::reset_cache() {
    // KV state is maintained by the shared engine cache (StaticKVCache).
}

static void dump_tensor_brief_append(const infinicore::Tensor &t, const char *name, const char *path) {
    if (!path) return;
    try {
        auto cpu_t = t->to(infinicore::Device::cpu());
        const auto &shp = cpu_t->shape();
        const auto dt = cpu_t->dtype();
        std::ofstream f(path, std::ios::app);
        if (!f) return;
        f << name << " shape=[";
        for (size_t i = 0; i < shp.size(); ++i) {
            if (i) f << ",";
            f << shp[i];
        }
        f << "] dtype=" << static_cast<int>(dt) << "\n";

        const size_t n = cpu_t->numel();
        const size_t k = std::min<size_t>(n, 16);
        std::vector<float> buf(k);
        if (dt == infinicore::DataType::BF16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < k; ++i) {
                uint32_t u = static_cast<uint32_t>(p[i]) << 16;
                buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < k; ++i) {
                uint32_t u = (p[i] & 0x8000) << 16 | ((p[i] & 0x7fff) + (127 - 15)) << 23 | (p[i] & 0x03ff) << 13;
                buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F32) {
            const float *p = reinterpret_cast<const float *>(cpu_t->data());
            for (size_t i = 0; i < k; ++i) buf[i] = p[i];
        } else {
            f << "  (brief dump skipped for dtype)\n";
            return;
        }
        f << "  first[" << k << "]:";
        for (size_t i = 0; i < k; ++i) f << " " << buf[i];
        f << "\n";
    } catch (...) {
    }
}

static void dump_tensor_brief_tail_append(const infinicore::Tensor &t, const char *name, const char *path) {
    if (!path) return;
    try {
        auto cpu_t = t->to(infinicore::Device::cpu());
        const auto &shp = cpu_t->shape();
        const auto dt = cpu_t->dtype();
        std::ofstream f(path, std::ios::app);
        if (!f) return;
        f << name << " shape=[";
        for (size_t i = 0; i < shp.size(); ++i) {
            if (i) f << ",";
            f << shp[i];
        }
        f << "] dtype=" << static_cast<int>(dt) << "\n";

        const size_t n = cpu_t->numel();
        const size_t k = std::min<size_t>(n, 16);
        std::vector<float> buf(k);
        if (dt == infinicore::DataType::BF16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < k; ++i) {
                uint32_t u = static_cast<uint32_t>(p[n - k + i]) << 16;
                buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < k; ++i) {
                uint32_t u = (p[n - k + i] & 0x8000) << 16 | ((p[n - k + i] & 0x7fff) + (127 - 15)) << 23 |
                             (p[n - k + i] & 0x03ff) << 13;
                buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F32) {
            const float *p = reinterpret_cast<const float *>(cpu_t->data());
            for (size_t i = 0; i < k; ++i) buf[i] = p[n - k + i];
        } else {
            f << "  (tail dump skipped for dtype)\n";
            return;
        }

        f << "  tail[" << k << "]:";
        for (size_t i = 0; i < k; ++i) f << " " << buf[i];
        f << "\n";
    } catch (...) {
    }
}

infinicore::Tensor MiniCPMSALAAttention::forward(const infinicore::Tensor &hidden_states,
                                                 const infinicore::Tensor &position_ids,
                                                 std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                                 std::optional<infinicore::Tensor> past_sequence_lengths,
                                                 std::optional<infinicore::Tensor> total_sequence_lengths,
                                                 std::optional<infinicore::Tensor> input_offsets,
                                                 std::optional<infinicore::Tensor> cu_seqlens,
                                                 std::optional<infinicore::Tensor> block_tables,
                                                 std::optional<infinicore::Tensor> slot_mapping) const {
    (void)input_offsets;
    (void)block_tables;
    (void)slot_mapping;
    return forward_dense_(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, cu_seqlens);
}

infinicore::Tensor MiniCPMSALAAttention::forward_dense_(const infinicore::Tensor &hidden_states,
                                                       const infinicore::Tensor &position_ids,
                                                       std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                                       std::optional<infinicore::Tensor> past_sequence_lengths,
                                                       std::optional<infinicore::Tensor> total_sequence_lengths,
                                                       std::optional<infinicore::Tensor> cu_seqlens) const {
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
            throw std::runtime_error("MiniCPMSALAAttention: rotary_emb is not set but use_rope=true");
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
            throw std::runtime_error("MiniCPMSALAAttention: Unexpected position_ids shape");
        }

        rotary_emb_->forward(q_reshaped, pos_ids_for_rope, true);
        rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);
    }

    // Compute dense attention (GQA): reshape as LlamaAttention does
    size_t total_seq_len = seq_len;
    size_t cache_pos = 0;
    const bool has_cache_meta = past_sequence_lengths.has_value() && total_sequence_lengths.has_value();
    if (has_cache_meta) {
        // Single device-to-host sync: read both scalars (engine could pass these as scalars later).
        auto past_cpu = past_sequence_lengths.value()->to(infinicore::Device::cpu());
        auto total_cpu = total_sequence_lengths.value()->to(infinicore::Device::cpu());
        cache_pos = reinterpret_cast<int32_t *>(past_cpu->data())[0];
        size_t total_seq_len_raw = reinterpret_cast<int32_t *>(total_cpu->data())[0];
        total_seq_len = total_seq_len_raw;
        // Some engine call sites pass `total_sequence_lengths` as the *input* length (e.g. 1 for decode),
        // while `past_sequence_lengths` is the cached KV length. Attention needs total KV length.
        // Use KV semantics: total_kv_len = cache_pos + current seq_len.
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

    // HF-like dense KV caching using the engine-provided StaticKVCache.
    infinicore::Tensor k_total = k_permuted;
    infinicore::Tensor v_total = v_permuted;
    std::shared_ptr<cache::StaticKVCache> static_kv_cache = nullptr;
    if (kv_cache != nullptr && has_cache_meta) {
        static_kv_cache = std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache);
        if (!static_kv_cache) {
            throw std::runtime_error("MiniCPMSALAAttention: Unsupported cache type (expected StaticKVCache)");
        }
        // Default behavior: update cache here. For minicpm4 decode we may override and let InfLLM-v2 update.
        auto [k_cached, v_cached] = static_kv_cache->update(
            cache_layer_idx_, k_permuted, v_permuted, past_sequence_lengths.value());
        k_total = k_cached;
        v_total = v_cached;
    } else {
        // No cache metadata => treat as prefill-only.
        total_seq_len = seq_len;
    }

    // Slice to total_seq_len (decode-only / cont-batch)
    if (total_seq_len > k_total->shape()[2]) {
        throw std::runtime_error("MiniCPMSALAAttention: total_seq_len exceeds available KV length (cache not correctly updated)");
    }
    k_total = k_total->narrow({{2, 0, total_seq_len}});
    v_total = v_total->narrow({{2, 0, total_seq_len}});

    // Debug KV cache parity: dump brief k/v for layer0 when enabled.
    // Helps verify that decode cached KV matches full-sequence KV.
    {
        const char *kv_prefix = std::getenv("INFINI_DEBUG_KV_DUMP_PREFIX");
        if (kv_prefix && kv_prefix[0] != '\0' && kv_prefix[0] != '0' && layer_idx_ == 0 && batch_size == 1) {
            std::string path = std::string("/tmp/kv_dump_") + kv_prefix + ".txt";
            char namek[256];
            char namev[256];
            std::snprintf(namek, sizeof(namek), "k_total cache_pos=%zu total_seq_len=%zu seq_len=%zu", cache_pos, total_seq_len, seq_len);
            std::snprintf(namev, sizeof(namev), "v_total cache_pos=%zu total_seq_len=%zu seq_len=%zu", cache_pos, total_seq_len, seq_len);
            // Ensure cache update kernels are finished before dumping to CPU.
            infinicore::context::syncStream();
            dump_tensor_brief_append(k_total, namek, path.c_str());
            dump_tensor_brief_append(v_total, namev, path.c_str());
            dump_tensor_brief_tail_append(k_total, namek, path.c_str());
            dump_tensor_brief_tail_append(v_total, namev, path.c_str());
        }
    }

    infinicore::Tensor attn_output;
    if (!is_sparse_layer_) {
        // Lightning-attn: Simple GLA (HF-aligned), same as test/infinicore/ops/gla_attention.py.
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

        // Lightning GLA decode must use recurrent state (StaticKVCache) whenever available.
        const bool is_lightning_decode = has_cache_meta && static_kv_cache && (seq_len < total_seq_len);
        if (is_lightning_decode && !static_kv_cache->has_gla_recurrent_state()) {
            throw std::runtime_error(
                "MiniCPMSALAAttention(lightning): Lightning decode requires StaticKVCache gla_recurrent_state "
                "(missing recurrent buffer in StaticKVCache).");
        }

        const bool recurrent_gla = static_kv_cache && static_kv_cache->has_gla_recurrent_state() && has_cache_meta;

        infinicore::Tensor gla_out;
        if (recurrent_gla && seq_len == 1 && total_seq_len > 1) {
            auto S = static_kv_cache->gla_recurrent_state_for_layer(cache_layer_idx_);
            auto q_new = q_bthd;
            auto k_new = k_bthd->narrow({{1, total_seq_len - 1, 1}});
            auto v_new = v_bthd->narrow({{1, total_seq_len - 1, 1}});
            gla_out = infinicore::op::simple_gla_decode_step(q_new, k_new, v_new, S, g_gamma_, scaling_);
        } else {
            infinicore::Tensor q_full;
            if (seq_len == total_seq_len) {
                q_full = q_bthd;
            } else {
                // Decode: q has seq_len (e.g. 1), kv has total_seq_len; pad q to [B, total_seq_len, H, D].
                q_full = infinicore::Tensor::zeros(
                    {batch_size, total_seq_len, n_h, head_dim_}, q_bthd->dtype(), q_bthd->device());
                auto q_slot = q_full->narrow({{1, total_seq_len - seq_len, seq_len}});
                q_slot->copy_from(q_bthd);
            }
            // Fused prefill: naive kernel for head_dim<=64; chunked/tiled kernel for head_dim>64 (e.g. 128).
            bool use_fused_prefill = (batch_size == 1) && (seq_len == total_seq_len);
            if (use_fused_prefill) {
                gla_out = infinicore::op::simple_gla_prefill(q_full, k_bthd, v_bthd, g_gamma_, scaling_);
            } else {
                gla_out = infinicore::op::simple_gla_attention(q_full, k_bthd, v_bthd, g_gamma_, scaling_);
            }

            // Keep per-layer recurrent state aligned with simple_gla_attention / prefill outputs.
            // Use batched GEMM (CUDA+ATen) instead of O(seq_len) decode_step launches; see
            // simple_gla_recurrent_state_append_segment (closed form: S <- g^L S + Σ g^{L-1-j} outer(k,v)).
            if (recurrent_gla) {
                auto S = static_kv_cache->gla_recurrent_state_for_layer(cache_layer_idx_);
                if (cache_pos == 0) {
                    infinicore::op::zeros_(S);
                }
                auto k_seg = k_bthd->narrow({{1, cache_pos, seq_len}});
                auto v_seg = v_bthd->narrow({{1, cache_pos, seq_len}});
                infinicore::op::simple_gla_recurrent_state_append_segment(S, k_seg, v_seg, g_gamma_);
            }
        }

        infinicore::Tensor out_slice = (recurrent_gla && seq_len == 1 && total_seq_len > 1)
                                           ? gla_out
                                           : gla_out->narrow({{1, total_seq_len - seq_len, seq_len}});
        attn_output = out_slice->view({batch_size, seq_len, n_h * head_dim_});
    } else {
        // minicpm4 layers must use InfLLM-v2 attention (hard error if not available).
        // NOTE: Lightning layers keep Simple GLA for correctness; only minicpm4 routes here.
        try {
            if (!total_sequence_lengths.has_value()) {
                throw std::runtime_error(
                    "MiniCPMSALAAttention(minicpm4): total_sequence_lengths is required for InfLLM-v2 path");
            }
            // `infllmv2_kvcache` expects the number of valid K/V entries in the
            // provided cache tensors. Since we already appended the current
            // token via StaticKVCache::update, the valid length is the total
            // KV length (past + current token).
            const auto cache_lens = total_sequence_lengths.value();

            // Prefill: InfLLM-v2 varlen (Q and K packed lengths match `seq_len == total_seq_len` here).
            // Decode: `seq_len < total_seq_len` — use `infllmv2_kvcache` after StaticKVCache::update
            // (valid KV length == `total_seq_len`). Using varlen for decode (1 query vs long K) hit NaNs
            // in practice for modest sequence lengths; kvcache matches operator tests and Flash path.
            const bool force_varlen_decode = [&]() {
                const char *env = std::getenv("INFINI_MINICPM4_DECODE_VARLEN");
                return env && env[0] != '\0' && env[0] != '0';
            }();

            if (seq_len == total_seq_len || (force_varlen_decode && batch_size == 1)) {
                if (batch_size != 1) {
                    throw std::runtime_error("MiniCPMSALAAttention(minicpm4): varlen prefill path currently requires batch_size=1");
                }
                auto q_bshd = q_reshaped->contiguous();                     // [B, S, n_h, D]
                auto k_btkd = k_total->permute({0, 2, 1, 3})->contiguous();  // [B, T, n_kv, D]
                auto v_btkd = v_total->permute({0, 2, 1, 3})->contiguous();  // [B, T, n_kv, D]
                auto q_var = q_bshd->view({static_cast<ptrdiff_t>(seq_len), static_cast<ptrdiff_t>(num_attention_heads_), static_cast<ptrdiff_t>(head_dim_)});
                auto k_var = k_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});
                auto v_var = v_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});

                auto cuq_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
                reinterpret_cast<int32_t *>(cuq_cpu->data())[0] = 0;
                reinterpret_cast<int32_t *>(cuq_cpu->data())[1] = static_cast<int32_t>(seq_len);
                infinicore::Tensor cu_q = cuq_cpu->to(q_var->device());
                // cu_k corresponds to the full KV length used by k_var/v_var.
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
            } else if (static_kv_cache) {
                if (batch_size != 1) {
                    throw std::runtime_error("MiniCPMSALAAttention(minicpm4): kvcache decode path currently requires batch_size=1");
                }
                auto q_bshd = q_reshaped->contiguous();                     // [B, S_q, n_h, D]
                auto k_bthd = k_total->permute({0, 2, 1, 3})->contiguous(); // [B, T, n_kv, D]
                auto v_bthd = v_total->permute({0, 2, 1, 3})->contiguous(); // [B, T, n_kv, D]

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
                attn_output = out_bshd->contiguous()->view(
                    {batch_size, seq_len, num_attention_heads_ * head_dim_});
            } else {
                throw std::runtime_error(
                    "MiniCPMSALAAttention(minicpm4): decode requires StaticKVCache (missing cache metadata or cache)");
            }
        } catch (const std::exception &e) {
            throw std::runtime_error(
                std::string("MiniCPMSALAAttention(minicpm4): InfLLM-v2 attention failed. ")
                + "This build must provide InfLLM-v2 (ENABLE_INFLLMV2+ENABLE_ATEN) and the infllmv2_cuda_impl .so "
                + "must be available via LD_PRELOAD/LD_LIBRARY_PATH. Original error: " + e.what());
        }
    }

    // Output norm + gate variants
    if (use_output_gate_) {
        if (o_gate_) {
            // Sparse (minicpm4): y = sigmoid(o_gate(x)) * attn_output
            auto gate_in = hidden_states;
            auto gate = o_gate_->forward(gate_in);
            infinicore::op::sigmoid_(gate, gate);
            attn_output = infinicore::op::mul(attn_output, gate);
        } else if (z_proj_) {
            // Lightning: match HF LightningAttention: o_norm(o) then o * sigmoid(z_proj(x)).
            auto z_in = hidden_states;
            auto z = z_proj_->forward(z_in);
            infinicore::op::sigmoid_(z, z);
            if (use_output_norm_ && o_norm_) {
                attn_output = o_norm_->forward(attn_output);
            }
            attn_output = infinicore::op::mul(attn_output, z);
        }
    } else if (use_output_norm_ && o_norm_) {
        attn_output = o_norm_->forward(attn_output);
    }

    auto attn_out_mut = attn_output;
    auto out = o_proj_->forward(attn_out_mut);

    return out;
}

} // namespace infinilm::models::minicpm_sala
