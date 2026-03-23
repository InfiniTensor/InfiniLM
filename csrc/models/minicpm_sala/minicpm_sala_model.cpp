#include "minicpm_sala_model.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>

namespace infinilm::models::minicpm_sala {

namespace {

void dump_tensor_to_bin_if_enabled(const infinicore::Tensor &tensor, const char *bin_path) {
    if (!bin_path || !std::getenv("INFINI_DEBUG_ATTN_DUMP")) return;
    try {
        // Debug-only: make sure any pending GPU work for `tensor` is finished
        // before reading/copying values to host. This avoids stale/uninitialized
        // dumps that can look like all-zeros.
        infinicore::context::syncStream();
        auto cpu_t = tensor->to(infinicore::Device::cpu());
        const size_t n = cpu_t->numel();
        const auto dt = cpu_t->dtype();
        std::vector<float> f32_buf(n);
        if (dt == infinicore::DataType::BF16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = static_cast<uint32_t>(p[i]) << 16;
                f32_buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F32) {
            const float *p = reinterpret_cast<const float *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) f32_buf[i] = p[i];
        } else if (dt == infinicore::DataType::F16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = (p[i] & 0x8000) << 16 | ((p[i] & 0x7fff) + (127 - 15)) << 23 | (p[i] & 0x03ff) << 13;
                f32_buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else return;
        std::ofstream bin(bin_path, std::ios::binary);
        if (bin) bin.write(reinterpret_cast<const char *>(f32_buf.data()), n * sizeof(float));
    } catch (...) {}
}

void log_tensor_stats_if_enabled(const infinicore::Tensor &tensor,
                                 const char *hypothesis_id,
                                 const char *location,
                                 const char *message) {
    const char *log_path = std::getenv("INFINI_DEBUG_LOG");
    if (!log_path) {
        return;
    }
    try {
        auto cpu_t = tensor->to(infinicore::Device::cpu());
        const size_t n = cpu_t->numel();
        const auto &shp = cpu_t->shape();
        const auto dt = cpu_t->dtype();
        std::vector<float> f32_buf(n);
        if (dt == infinicore::DataType::BF16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = static_cast<uint32_t>(p[i]) << 16;
                f32_buf[i] = *reinterpret_cast<float *>(&u);
            }
        } else if (dt == infinicore::DataType::F32) {
            const float *p = reinterpret_cast<const float *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) f32_buf[i] = p[i];
        } else if (dt == infinicore::DataType::F16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(cpu_t->data());
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = (p[i] & 0x8000) << 16 | ((p[i] & 0x7fff) + (127 - 15)) << 23 | (p[i] & 0x03ff) << 13;
                f32_buf[i] = *reinterpret_cast<float *>(&u);
            }
        }
        float mn = f32_buf.empty() ? 0.f : f32_buf[0];
        float mx = mn;
        double sum = 0.0;
        double ss = 0.0;
        for (float v : f32_buf) {
            mn = std::min(mn, v);
            mx = std::max(mx, v);
            sum += v;
            ss += static_cast<double>(v) * static_cast<double>(v);
        }
        const double mean = n ? (sum / static_cast<double>(n)) : 0.0;
        const double norm = ss > 0.0 ? std::sqrt(ss) : 0.0;

        std::ofstream log(log_path, std::ios::app);
        if (log) {
            std::string shape_json = "[";
            for (size_t i = 0; i < shp.size(); ++i) {
                shape_json += (i ? "," : "") + std::to_string(shp[i]);
            }
            shape_json += "]";
            const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                    std::chrono::system_clock::now().time_since_epoch())
                                    .count();
            log << "{\"sessionId\":\"9146ea\",\"hypothesisId\":\"" << hypothesis_id
                << "\",\"location\":\"" << location
                << "\",\"message\":\"" << message
                << "\",\"data\":{\"shape\":" << shape_json
                << ",\"min\":" << mn
                << ",\"max\":" << mx
                << ",\"mean\":" << mean
                << ",\"l2\":" << norm
                << "},\"timestamp\":" << now_ms << "}\n";
        }
    } catch (...) {
        // Best-effort diagnostics; never throw from logging path.
    }
}

} // namespace

MiniCPMSALAModel::MiniCPMSALAModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info,
                                   backends::AttentionBackend attention_backend)
    : model_config_(std::move(model_config)),
      rank_info_(rank_info),
      attention_backend_(attention_backend) {

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config_->get_dtype();
    compute_device_ = device;

    hidden_size_ = model_config_->get<size_t>("hidden_size");
    dim_model_base_ = model_config_->get_or<double>("dim_model_base", static_cast<double>(hidden_size_));
    scale_emb_ = model_config_->get_or<double>("scale_emb", 1.0);

    const size_t vocab_size = model_config_->get<size_t>("vocab_size");
    const size_t num_layers = model_config_->get<size_t>("num_hidden_layers");

    INFINICORE_NN_MODULE_INIT(embed_tokens, vocab_size, hidden_size_, std::nullopt, dtype, device);
    INFINICORE_NN_MODULE_INIT(norm, hidden_size_, model_config_->get<double>("rms_norm_eps"), dtype, device);

    // Shared rotary embedding (used by lightning layers only)
    INFINICORE_NN_MODULE_INIT(rotary_emb,
                              model_config_->get_head_dim(),
                              model_config_->get<size_t>("max_position_embeddings"),
                              model_config_->get<double>("rope_theta"),
                              infinicore::nn::RoPE::Algo::GPT_NEOX,
                              dtype,
                              device,
                              model_config_->get_rope_scaling());

    // Mixer types per-layer decide attention flavor (minicpm4 vs lightning-attn).
    std::vector<std::string> mixer_types;
    try {
        mixer_types = model_config_->get<std::vector<std::string>>("mixer_types");
    } catch (...) {
        mixer_types.assign(num_layers, "minicpm4");
    }
    if (mixer_types.size() != num_layers) {
        mixer_types.resize(num_layers, mixer_types.empty() ? "minicpm4" : mixer_types.back());
    }
    mixer_types_ = mixer_types;

    layers_.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        layers_.push_back(this->register_module<MiniCPMSALADecoderLayer>(
            "layers." + std::to_string(i), model_config_, device, i, mixer_types[i], rank_info_, attention_backend_));
        layers_.back()->set_rotary_emb(rotary_emb_);
    }
}

void MiniCPMSALAModel::reset_cache(const cache::CacheConfig *cache_config) {
    if (cache_config == nullptr) {
        kv_cache_minicpm4_ = nullptr;
        kv_cache_lightning_ = nullptr;
        for (auto &layer : layers_) {
            layer->reset_cache();
        }
        return;
    }

    if (auto static_cfg = dynamic_cast<const cache::StaticKVCacheConfig *>(cache_config)) {
        // Allocate separate caches by KV shape to avoid per-layer padding copies.
        const size_t num_hidden_layers = model_config_->get<size_t>("num_hidden_layers");
        // mixer_types_ is filled in ctor from model_config_->get("mixer_types").
        const size_t minicpm4_layer_count =
            !mixer_types_.empty() ? std::count(mixer_types_.begin(), mixer_types_.end(), "minicpm4") : num_hidden_layers;
        const size_t lightning_layer_count = num_hidden_layers - minicpm4_layer_count;

        const size_t base_kv_heads = model_config_->get<size_t>("num_key_value_heads");
        const size_t base_head_dim = model_config_->get<size_t>("head_dim");
        const size_t lightning_kv_heads = model_config_->get_or<size_t>("lightning_nkv", base_kv_heads);
        const size_t lightning_head_dim = model_config_->get_or<size_t>("lightning_head_dim", base_head_dim);
        const size_t lightning_nh = model_config_->get_or<size_t>("lightning_nh", model_config_->get<size_t>("num_attention_heads"));
        const int tp_sz = std::max(1, rank_info_.tp_size);
        const size_t lightning_nh_rank = lightning_nh / static_cast<size_t>(tp_sz);

        kv_cache_minicpm4_ = (minicpm4_layer_count > 0)
                                 ? std::make_shared<cache::StaticKVCache>(
                                       /*k_dim=*/base_head_dim,
                                       /*v_dim=*/base_head_dim,
                                       /*num_k_heads=*/base_kv_heads,
                                       /*num_v_heads=*/base_kv_heads,
                                       /*num_layers=*/minicpm4_layer_count,
                                       /*max_positional_embedding=*/model_config_->get<size_t>("max_position_embeddings"),
                                       /*dtype=*/model_config_->get_dtype(),
                                       *static_cfg,
                                       rank_info_)
                                 : nullptr;

        kv_cache_lightning_ = (lightning_layer_count > 0)
                                   ? std::make_shared<cache::StaticKVCache>(
                                         /*k_dim=*/lightning_head_dim,
                                         /*v_dim=*/lightning_head_dim,
                                         /*num_k_heads=*/lightning_kv_heads,
                                         /*num_v_heads=*/lightning_kv_heads,
                                         /*num_layers=*/lightning_layer_count,
                                         /*max_positional_embedding=*/model_config_->get<size_t>("max_position_embeddings"),
                                         /*dtype=*/model_config_->get_dtype(),
                                         *static_cfg,
                                         rank_info_,
                                         /*gla_recurrent_num_heads=*/lightning_nh_rank,
                                         /*gla_recurrent_head_dim=*/lightning_head_dim)
                                 : nullptr;
    } else {
        // This refactor implements HF-like dense caching only.
        throw std::runtime_error("MiniCPMSALAModel::reset_cache: Unsupported cache type (expected StaticKVCacheConfig)");
    }

    for (auto &layer : layers_) {
        layer->reset_cache();
    }
}

infinicore::Tensor MiniCPMSALAModel::forward(const infinicore::Tensor &input_ids,
                                             const infinicore::Tensor &position_ids,
                                             std::optional<infinicore::Tensor> past_sequence_lengths,
                                             std::optional<infinicore::Tensor> total_sequence_lengths,
                                             std::optional<infinicore::Tensor> input_offsets,
                                             std::optional<infinicore::Tensor> cu_seqlens,
                                             std::optional<infinicore::Tensor> block_tables,
                                             std::optional<infinicore::Tensor> slot_mapping) const {
    // MuP scaling baked into weights at load time for minicpm_sala; no forward scaling here.
    auto hs = embed_tokens_->forward(input_ids);
    if (std::getenv("INFINI_DEBUG_LOG")) {
        infinicore::context::syncDevice();
    }
    log_tensor_stats_if_enabled(hs,
                                "INF_C",
                                "minicpm_sala_model.cpp:embed_tokens",
                                "Inf embed output");
    dump_tensor_to_bin_if_enabled(hs, "/tmp/inf_embed_out.bin");

    for (size_t i = 0; i < layers_.size(); ++i) {
        std::shared_ptr<cache::Cache> layer_cache;
        if (!mixer_types_.empty() && mixer_types_[i] == "minicpm4") {
            layer_cache = kv_cache_minicpm4_;
        } else {
            layer_cache = kv_cache_lightning_;
        }
        hs = layers_[i]->forward(hs,
                                 position_ids,
                                 layer_cache,
                                 past_sequence_lengths,
                                 total_sequence_lengths,
                                 input_offsets,
                                 cu_seqlens,
                                 block_tables,
                                 slot_mapping);
        if (const char *env = std::getenv("MINICPM_SALA_LAYER_TRACE")) {
            if (env[0] != '\0' && env[0] != '0') {
                fprintf(stderr, "[minicpm_sala][layer_trace] layer=%zu mixer=%s\n",
                        i,
                        mixer_types_.empty() ? "unknown" : mixer_types_[i].c_str());
                fflush(stderr);
            }
        }
    }

    hs = norm_->forward(hs);
    // Debug-only: help locate where non-finite values first appear
    // during decode (final norm before lm_head).
    if (std::getenv("INFINI_DEBUG_LOG")) {
        log_tensor_stats_if_enabled(hs,
                                    "INF_H",
                                    "minicpm_sala_model.cpp:final_norm",
                                    "Inf final hidden after norm");
    }
    dump_tensor_to_bin_if_enabled(hs, "/tmp/inf_final_hidden.bin");
    return hs;
}

} // namespace infinilm::models::minicpm_sala
