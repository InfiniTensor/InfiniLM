#include "minicpm_sala_decoder_layer.hpp"

#include "infinicore/ops.hpp"
#include "infinicore/context/context.hpp"
#include <cmath>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace infinilm::models::minicpm_sala {

namespace {

void log_tensor_stats_if_enabled(const infinicore::Tensor &tensor,
                                 size_t layer_idx,
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
                << "\",\"data\":{\"layer\":" << layer_idx
                << ",\"shape\":" << shape_json
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

// Convert to float and optionally dump to binary (for alignment checks).
void tensor_to_f32_and_dump(const infinicore::Tensor &tensor, const char *bin_path) {
    if (!bin_path || !std::getenv("INFINI_DEBUG_ATTN_DUMP")) return;
    try {
        // Debug-only: ensure pending GPU work for `tensor` is complete
        // before CPU copy so the binary dump reflects real values.
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

} // namespace

MiniCPMSALADecoderLayer::MiniCPMSALADecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                                 const infinicore::Device &device,
                                                 size_t layer_idx,
                                                 const std::string &mixer_type,
                                                 engine::distributed::RankInfo rank_info,
                                                 backends::AttentionBackend attention_backend) {
    layer_idx_ = layer_idx;
    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    const double eps = model_config->get<double>("rms_norm_eps");

    // MuP residual scaling at forward (o_proj/down_proj not scaled in loader for minicpm_sala).
    const double scale_depth = model_config->get_or<double>("scale_depth", 1.0);
    const size_t num_layers = model_config->get<size_t>("num_hidden_layers");
    residual_scale_ = scale_depth / std::sqrt(static_cast<double>(num_layers));

    INFINICORE_NN_MODULE_INIT(input_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(self_attn, model_config, device, layer_idx, mixer_type, rank_info, attention_backend);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, model_config->get<size_t>("hidden_size"), eps, dtype, device);
    INFINICORE_NN_MODULE_INIT(mlp, model_config, device);
}

void MiniCPMSALADecoderLayer::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    self_attn_->set_rotary_emb(rotary_emb);
}

void MiniCPMSALADecoderLayer::reset_cache() {
    self_attn_->reset_cache();
}

infinicore::Tensor MiniCPMSALADecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                                    const infinicore::Tensor &position_ids,
                                                    std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                                    std::optional<infinicore::Tensor> past_sequence_lengths,
                                                    std::optional<infinicore::Tensor> total_sequence_lengths,
                                                    std::optional<infinicore::Tensor> input_offsets,
                                                    std::optional<infinicore::Tensor> cu_seqlens,
                                                    std::optional<infinicore::Tensor> block_tables,
                                                    std::optional<infinicore::Tensor> slot_mapping) const {
    // Pre-norm attention
    auto hs1 = input_layernorm_->forward(hidden_states);
    if (layer_idx_ == 0)
        tensor_to_f32_and_dump(hs1, "/tmp/inf_layer0_attn_input.bin");

    const bool log_all_layers = []() {
        const char *env = std::getenv("INFINI_DEBUG_LOG_ALL_LAYERS");
        if (!env) return false;
        return env[0] != '\0' && env[0] != '0';
    }();

    if (log_all_layers || layer_idx_ < 2) {
        log_tensor_stats_if_enabled(hs1,
                                    layer_idx_,
                                    "INF_B",
                                    "minicpm_sala_decoder_layer.cpp:input_layernorm",
                                    "Inf input layernorm output");
    }
    auto attn_out = self_attn_->forward(
        hs1,
        position_ids,
        kv_cache,
        past_sequence_lengths,
        total_sequence_lengths,
        input_offsets,
        cu_seqlens,
        block_tables,
        slot_mapping);

    // residual + scale_down * attn_out (MuP)
    auto ones_attn = infinicore::Tensor::empty(attn_out->shape(), attn_out->dtype(), attn_out->device());
    infinicore::op::ones_(ones_attn);
    auto out1 = infinicore::op::addcmul(hidden_states, attn_out, ones_attn, static_cast<float>(residual_scale_));

    // Pre-norm MLP
    auto hs2 = post_attention_layernorm_->forward(out1);
    auto mlp_out = mlp_->forward(hs2);
    // residual + scale_down * mlp_out (MuP)
    auto ones_mlp = infinicore::Tensor::empty(mlp_out->shape(), mlp_out->dtype(), mlp_out->device());
    infinicore::op::ones_(ones_mlp);
    auto out2 = infinicore::op::addcmul(out1, mlp_out, ones_mlp, static_cast<float>(residual_scale_));

    // #region agent log
    if (log_all_layers || layer_idx_ < 3) {
        log_tensor_stats_if_enabled(out2,
                                    layer_idx_,
                                    "INF_L",
                                    "minicpm_sala_decoder_layer.cpp:forward_output",
                                    "Inf decoder layer output stats");
        if (std::getenv("INFINI_DEBUG_ATTN_DUMP")) {
            char path[64];
            std::snprintf(path, sizeof(path), "/tmp/inf_layer_out_%zu.bin", layer_idx_);
            tensor_to_f32_and_dump(out2, path);
        }
    }
    // #endregion

    return out2;
}

} // namespace infinilm::models::minicpm_sala
