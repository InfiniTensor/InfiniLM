#include "minicpm_sala_attention.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/ops/infllmv2_attention.hpp"
#include "infinicore/ops/simple_gla_attention.hpp"
#include "infinicore/ops/simple_gla_prefill.hpp"
#include "../debug_utils/tensor_utils.hpp"

#include <cmath>
#include <cstdlib>
#include <dlfcn.h>
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

inline void log_vram_trace_if_enabled(const char *tag,
                                      size_t layer_idx,
                                      size_t cache_pos,
                                      size_t total_seq_len,
                                      size_t seq_len) {
    const char *env = std::getenv("MINICPM_SALA_VRAM_TRACE");
    if (!env || env[0] == '\0' || env[0] == '0') return;
    static bool banner_printed = false;
    if (!banner_printed) {
        banner_printed = true;
        fprintf(stderr, "[minicpm_sala][vram] tracing enabled (MINICPM_SALA_VRAM_TRACE=%s)\n", env);
        fflush(stderr);
    }

    // Query VRAM via CUDA driver API without CUDA headers/toolkit.
    using CUresult = int;
    using CUdevice = int;
    using CUcontext = void *;
    using size_t_ = size_t;
    constexpr CUresult CUDA_SUCCESS = 0;

    using PFN_cuInit = CUresult (*)(unsigned int);
    using PFN_cuMemGetInfo = CUresult (*)(size_t_ *, size_t_ *);
    using PFN_cuGetErrorString = CUresult (*)(CUresult, const char **);

    static void *lib = nullptr;
    static PFN_cuInit p_cuInit = nullptr;
    static PFN_cuMemGetInfo p_cuMemGetInfo = nullptr;
    static PFN_cuGetErrorString p_cuGetErrorString = nullptr;
    static bool init_attempted = false;
    static bool init_ok = false;

    if (!init_attempted) {
        init_attempted = true;
        lib = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        if (lib) {
            p_cuInit = reinterpret_cast<PFN_cuInit>(dlsym(lib, "cuInit"));
            p_cuMemGetInfo = reinterpret_cast<PFN_cuMemGetInfo>(dlsym(lib, "cuMemGetInfo_v2"));
            if (!p_cuMemGetInfo) {
                p_cuMemGetInfo = reinterpret_cast<PFN_cuMemGetInfo>(dlsym(lib, "cuMemGetInfo"));
            }
            p_cuGetErrorString = reinterpret_cast<PFN_cuGetErrorString>(dlsym(lib, "cuGetErrorString"));
            if (p_cuInit && p_cuMemGetInfo) {
                CUresult st = p_cuInit(0);
                init_ok = (st == CUDA_SUCCESS);
                if (!init_ok && p_cuGetErrorString) {
                    const char *msg = nullptr;
                    p_cuGetErrorString(st, &msg);
                    fprintf(stderr, "[minicpm_sala][vram] cuInit failed: %s\n", msg ? msg : "unknown");
                    fflush(stderr);
                }
            }
        }
        if (!lib || !p_cuInit || !p_cuMemGetInfo) {
            fprintf(stderr, "[minicpm_sala][vram] could not resolve CUDA driver symbols (libcuda.so.1)\n");
            fflush(stderr);
        }
    }

    if (!init_ok || !p_cuMemGetInfo) return;
    size_t free_b = 0;
    size_t total_b = 0;
    CUresult st = p_cuMemGetInfo(&free_b, &total_b);
    if (st != CUDA_SUCCESS || total_b == 0) return;
    const size_t used_b = total_b - free_b;
    const double used_mib = static_cast<double>(used_b) / (1024.0 * 1024.0);
    const double free_mib = static_cast<double>(free_b) / (1024.0 * 1024.0);
    fprintf(stderr,
            "[minicpm_sala][vram] layer=%zu cache_pos=%zu total_seq_len=%zu seq_len=%zu tag=%s used_mib=%.1f free_mib=%.1f\n",
            layer_idx, cache_pos, total_seq_len, seq_len, (tag ? tag : "null"), used_mib, free_mib);
    fflush(stderr);
}
} // namespace

static void log_tensor_stats_to_file_if_enabled(const infinicore::Tensor &tensor,
                                                const char *tag,
                                                size_t layer_idx,
                                                size_t cache_pos,
                                                size_t total_seq_len,
                                                size_t seq_len) {
    const char *log_path = std::getenv("INFINI_DEBUG_LOG");
    if (!log_path || !tag) return;
    try {
        auto cpu_t = tensor->to(infinicore::Device::cpu());
        const size_t n = cpu_t->numel();
        const auto dt = cpu_t->dtype();
        const auto &shp = cpu_t->shape();

        std::ofstream f(log_path, std::ios::app);
        if (!f) return;
        f << "[minicpm_sala][dump] layer=" << layer_idx
          << " cache_pos=" << cache_pos
          << " total_seq_len=" << total_seq_len
          << " seq_len=" << seq_len
          << " tag=" << tag
          << " shape=[";
        for (size_t i = 0; i < shp.size(); ++i) {
            if (i) f << ",";
            f << shp[i];
        }
        f << "] dtype=" << static_cast<int>(dt) << " numel=" << n << "\n";

        if (n == 0) return;
        const size_t k = std::min<size_t>(n, 16);
        std::vector<float> vals(n);

        if (dt == infinicore::DataType::BF16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = static_cast<uint32_t>(p[i]) << 16;
                vals[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = (p[i] & 0x8000) << 16 | ((p[i] & 0x7fff) + (127 - 15)) << 23 | (p[i] & 0x03ff) << 13;
                vals[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F32) {
            const float *p = reinterpret_cast<const float *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) vals[i] = p[i];
        } else {
            f << "  (stats skipped for dtype)\n";
            return;
        }

        float mn = vals[0], mx = vals[0];
        double sum = 0.0;
        for (float v : vals) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += static_cast<double>(v);
        }
        f << "  min=" << mn << " max=" << mx << " mean=" << (sum / (double)n) << "\n";
        f << "  first[" << k << "]:";
        for (size_t i = 0; i < k; ++i) f << " " << vals[i];
        f << "\n";
        if (n > k) {
            f << "  last[" << k << "]:";
            for (size_t i = n - k; i < n; ++i) f << " " << vals[i];
            f << "\n";
        }
    } catch (...) {
    }
}

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
    } else {
        // Lightning layers have their own head config.
        num_attention_heads_ = model_config_->get_or<size_t>("lightning_nh", model_config_->get<size_t>("num_attention_heads"));
        num_key_value_heads_ = model_config_->get_or<size_t>("lightning_nkv", model_config_->get<size_t>("num_key_value_heads"));
        head_dim_ = model_config_->get_or<size_t>("lightning_head_dim", model_config_->get<size_t>("head_dim"));
    }
    scaling_ = static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_dim_)));

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
        if (const char *dump_env = std::getenv("MINICPM_SALA_DUMP_DECODE")) {
            if (dump_env[0] != '\0' && dump_env[0] != '0') {
                if (const char *log_path = std::getenv("INFINI_DEBUG_LOG")) {
                    try {
                        std::ofstream f(log_path, std::ios::app);
                        if (f) {
                            f << "[minicpm_sala][kv_len_fix] layer=" << layer_idx_
                              << " cache_pos=" << cache_pos
                              << " seq_len=" << seq_len
                              << " total_seq_len_raw=" << total_seq_len_raw
                              << " total_seq_len_used=" << total_seq_len
                              << "\n";
                        }
                    } catch (...) {
                    }
                }
            }
        }
    } else if (total_sequence_lengths.has_value()) {
        total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
    }

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
            layer_idx_, k_permuted, v_permuted, past_sequence_lengths.value());
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

    // Debug marker to confirm which attention path is used at decode.
    if (const char *dump_env = std::getenv("MINICPM_SALA_DUMP_DECODE")) {
        if (dump_env[0] != '\0' && dump_env[0] != '0') {
            if (const char *log_path = std::getenv("INFINI_DEBUG_LOG")) {
                try {
                    std::ofstream f(log_path, std::ios::app);
                    if (f) {
                        f << "[minicpm_sala][attn_enter] layer=" << layer_idx_
                          << " is_sparse=" << (is_sparse_layer_ ? 1 : 0)
                          << " has_cache_meta=" << (has_cache_meta ? 1 : 0)
                          << " kv_cache_null=" << (kv_cache == nullptr ? 1 : 0)
                          << " static_kv_cache_null=" << (static_kv_cache ? 0 : 1)
                          << " cache_pos=" << cache_pos
                          << " total_seq_len=" << total_seq_len
                          << " seq_len=" << seq_len
                          << "\n";
                    }
                } catch (...) {
                }
            }
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
        auto q_bthd = q_reshaped->contiguous(); // [B, S_q, H, D]
        auto k_bthd = k_use->permute({0, 2, 1, 3})->contiguous(); // [B, S_kv, H, D]
        auto v_bthd = v_use->permute({0, 2, 1, 3})->contiguous(); // [B, S_kv, H, D]
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
        infinicore::Tensor gla_out;
        // Fused prefill: naive kernel for head_dim<=64; chunked/tiled kernel for head_dim>64 (e.g. 128).
        const bool use_fused_prefill = (seq_len == total_seq_len && batch_size == 1);
        if (seq_len == total_seq_len && batch_size == 1) {
            const char *dbg = std::getenv("INFINI_DEBUG_GLA_PREFILL");
            if (dbg && dbg[0] != '\0' && dbg[0] != '0') {
                static bool logged_gla_prefill = false;
                if (!logged_gla_prefill) {
                    logged_gla_prefill = true;
                    if (use_fused_prefill) {
                        fprintf(stderr, "[minicpm_sala] GLA prefill: using simple_gla_prefill (head_dim=%zu)\n", static_cast<size_t>(head_dim_));
                    } else {
                        fprintf(stderr, "[minicpm_sala] GLA prefill: using simple_gla_attention (head_dim=%zu)\n", static_cast<size_t>(head_dim_));
                    }
                }
            }
        }
        if (use_fused_prefill) {
            // Prefill fast path: fused kernel (naive for D<=64, chunked for D>64).
            log_vram_trace_if_enabled("gla_prefill:before", layer_idx_, cache_pos, total_seq_len, seq_len);
            gla_out = infinicore::op::simple_gla_prefill(q_full, k_bthd, v_bthd, g_gamma_, scaling_);
            log_vram_trace_if_enabled("gla_prefill:after", layer_idx_, cache_pos, total_seq_len, seq_len);
        } else {
            gla_out = infinicore::op::simple_gla_attention(q_full, k_bthd, v_bthd, g_gamma_, scaling_);
        }
        // gla_out [B, total_seq_len, H, D]; take last seq_len positions.
        auto out_slice = gla_out->narrow({{1, total_seq_len - seq_len, seq_len}});
        attn_output = out_slice->view({batch_size, seq_len, n_h * head_dim_});
    } else {
        // minicpm4 layers must use InfLLM-v2 attention (hard error if not available).
        // NOTE: Lightning layers keep Simple GLA for correctness; only minicpm4 routes here.
        try {
            if (!total_sequence_lengths.has_value()) {
                throw std::runtime_error(
                    "MiniCPMSALAAttention(minicpm4): total_sequence_lengths is required for InfLLM-v2 path");
            }
            const auto cache_lens = total_sequence_lengths.value();

            // Prefill should use InfLLM-v2 varlen; decode should use kvcache.
            const bool is_prefill = (cache_pos == 0) && (seq_len == total_seq_len);

            if (seq_len == total_seq_len || !static_kv_cache) {
                // Prefill: use varlen attention over full prompt and update cache via StaticKVCache::update above.
                if (batch_size != 1) {
                    throw std::runtime_error("MiniCPMSALAAttention(minicpm4): varlen prefill path currently requires batch_size=1");
                }
                auto q_bshd = q_reshaped->contiguous();                     // [B, S, n_h, D]
                auto k_btkd = k_total->permute({0, 2, 1, 3})->contiguous();  // [B, T, n_kv, D]
                auto v_btkd = v_total->permute({0, 2, 1, 3})->contiguous();  // [B, T, n_kv, D]
                auto q_var = q_bshd->view({static_cast<ptrdiff_t>(seq_len), static_cast<ptrdiff_t>(num_attention_heads_), static_cast<ptrdiff_t>(head_dim_)});
                auto k_var = k_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});
                auto v_var = v_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});

                infinicore::Tensor cu_q = cu_seqlens.has_value() ? cu_seqlens.value() : infinicore::Tensor();
                infinicore::Tensor cu_k = cu_seqlens.has_value() ? cu_seqlens.value() : infinicore::Tensor();
                if (!cu_seqlens.has_value()) {
                    auto cuq_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
                    auto cuk_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
                    reinterpret_cast<int32_t *>(cuq_cpu->data())[0] = 0;
                    reinterpret_cast<int32_t *>(cuq_cpu->data())[1] = static_cast<int32_t>(seq_len);
                    reinterpret_cast<int32_t *>(cuk_cpu->data())[0] = 0;
                    reinterpret_cast<int32_t *>(cuk_cpu->data())[1] = static_cast<int32_t>(total_seq_len);
                    cu_q = cuq_cpu->to(q_var->device());
                    cu_k = cuk_cpu->to(q_var->device());
                }

                auto out_var = infinicore::op::infllmv2_varlen(
                    q_var, k_var, v_var,
                    cu_q, cu_k,
                    static_cast<int>(seq_len),
                    static_cast<int>(total_seq_len),
                    scaling_,
                    /*causal=*/true);
                attn_output = out_var->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
            } else {
                // Decode: use InfLLM-v2 varlen as a correctness-first path.
                // We construct a 1-token varlen query and attend over the full KV prefix.
                // This avoids potential issues in the kvcache kernel for certain GQA shapes.
                if (batch_size != 1) {
                    throw std::runtime_error("MiniCPMSALAAttention(minicpm4): varlen decode path currently requires batch_size=1");
                }
                auto q_bshd = q_reshaped->contiguous();                     // [B, S_q, n_h, D]
                auto k_btkd = k_total->permute({0, 2, 1, 3})->contiguous();  // [B, T, n_kv, D]
                auto v_btkd = v_total->permute({0, 2, 1, 3})->contiguous();  // [B, T, n_kv, D]
                auto q_var = q_bshd->view({static_cast<ptrdiff_t>(seq_len), static_cast<ptrdiff_t>(num_attention_heads_), static_cast<ptrdiff_t>(head_dim_)});
                auto k_var = k_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});
                auto v_var = v_btkd->view({static_cast<ptrdiff_t>(total_seq_len), static_cast<ptrdiff_t>(num_key_value_heads_), static_cast<ptrdiff_t>(head_dim_)});

                // cache_lens must be the current cache length BEFORE appending.
                if (!past_sequence_lengths.has_value()) {
                    throw std::runtime_error("MiniCPMSALAAttention(minicpm4): past_sequence_lengths is required for decode");
                }
                auto cache_lens_before = past_sequence_lengths.value();
                // Quick scalar dump of cache length (helps catch dtype/value issues).
                if (const char *dump_env = std::getenv("MINICPM_SALA_DUMP_DECODE")) {
                    if (dump_env[0] != '\0' && dump_env[0] != '0') {
                        if (const char *log_path = std::getenv("INFINI_DEBUG_LOG")) {
                            try {
                                auto cpu_lens = cache_lens_before->to(infinicore::Device::cpu());
                                int64_t v0 = 0;
                                if (cpu_lens->dtype() == infinicore::DataType::I32) {
                                    v0 = reinterpret_cast<const int32_t *>(cpu_lens->data())[0];
                                } else if (cpu_lens->dtype() == infinicore::DataType::I64) {
                                    v0 = reinterpret_cast<const int64_t *>(cpu_lens->data())[0];
                                }
                                std::ofstream f(log_path, std::ios::app);
                                if (f) {
                                    f << "[minicpm_sala][cache_lens_before] layer=" << layer_idx_
                                      << " cache_pos=" << cache_pos
                                      << " dtype=" << static_cast<int>(cpu_lens->dtype())
                                      << " v0=" << v0
                                      << " k_cache_T=" << (static_kv_cache ? std::get<0>(static_kv_cache->get_layer_kv_seq_major(layer_idx_))->shape()[1] : 0)
                                      << "\n";
                                }
                            } catch (...) {
                            }
                        }
                    }
                }

                // Optional debug dump for decode: enable with MINICPM_SALA_DUMP_DECODE=1
                // We dump only for early decode steps (small cache_pos) to keep logs small.
                if (const char *dump_env = std::getenv("MINICPM_SALA_DUMP_DECODE")) {
                    if (dump_env[0] != '\0' && dump_env[0] != '0') {
                        // Typical prompt lengths are small in sanity; dump the first few decode positions.
                        // Dump for all minicpm4 layers to avoid relying on layer index assumptions.
                        if (cache_pos <= 8) {
                            // Cache views from seq-major cache for debugging only.
                            auto [k_cache, v_cache] = static_kv_cache->get_layer_kv_seq_major(layer_idx_);
                            size_t tail_start = cache_pos > 2 ? cache_pos - 2 : 0;
                            size_t tail_len = std::min<size_t>(k_cache->shape()[1] - tail_start, 4);
                            auto k_tail = k_cache->narrow({{1, tail_start, tail_len}});
                            auto v_tail = v_cache->narrow({{1, tail_start, tail_len}});
                            auto k_pos = k_cache->narrow({{1, cache_pos, 1}});
                            auto v_pos = v_cache->narrow({{1, cache_pos, 1}});
                            auto k_prefix = k_cache->narrow({{1, 0, std::min<size_t>(k_cache->shape()[1], 5)}});
                            auto v_prefix = v_cache->narrow({{1, 0, std::min<size_t>(v_cache->shape()[1], 5)}});

                            // Marker line so we can confirm dump executed.
                            log_tensor_stats_to_file_if_enabled(hidden_states, "DUMP_TRIGGER", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(hidden_states, "hidden_states_in", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(q_bshd, "q_bshd", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(k_tail, "k_cache_tail_bthd", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(v_tail, "v_cache_tail_bthd", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(k_pos, "k_cache_pos_b1thd", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(v_pos, "v_cache_pos_b1thd", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(k_prefix, "k_cache_prefix5_b5thd", layer_idx_, cache_pos, total_seq_len, seq_len);
                            log_tensor_stats_to_file_if_enabled(v_prefix, "v_cache_prefix5_b5thd", layer_idx_, cache_pos, total_seq_len, seq_len);
                        }
                    }
                }

                infinicore::Tensor cu_q;
                infinicore::Tensor cu_k;
                // cu_seqlens for single batch: [0, seq_len] and [0, total_seq_len]
                auto cuq_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
                auto cuk_cpu = infinicore::Tensor::empty({2}, infinicore::DataType::I32, infinicore::Device::cpu());
                reinterpret_cast<int32_t *>(cuq_cpu->data())[0] = 0;
                reinterpret_cast<int32_t *>(cuq_cpu->data())[1] = static_cast<int32_t>(seq_len);
                reinterpret_cast<int32_t *>(cuk_cpu->data())[0] = 0;
                reinterpret_cast<int32_t *>(cuk_cpu->data())[1] = static_cast<int32_t>(total_seq_len);
                cu_q = cuq_cpu->to(q_var->device());
                cu_k = cuk_cpu->to(q_var->device());

                auto out_var = infinicore::op::infllmv2_varlen(
                    q_var, k_var, v_var, cu_q, cu_k,
                    static_cast<int>(seq_len),
                    static_cast<int>(total_seq_len),
                    scaling_,
                    /*causal=*/true);
                auto out_bshd = out_var->view({batch_size, seq_len, num_attention_heads_, head_dim_});
                if (const char *dump_env = std::getenv("MINICPM_SALA_DUMP_DECODE")) {
                    if (dump_env[0] != '\0' && dump_env[0] != '0') {
                        if (cache_pos <= 8) {
                            log_tensor_stats_to_file_if_enabled(out_bshd, "out_bshd", layer_idx_, cache_pos, total_seq_len, seq_len);
                        }
                    }
                }
                attn_output = out_bshd->contiguous()->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
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
