#include "minicpm_sala_attention.hpp"

#include "infinicore/ops.hpp"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <stdexcept>

namespace infinilm::models::minicpm_sala {

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
}

void MiniCPMSALAAttention::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    rotary_emb_ = rotary_emb;
}

void MiniCPMSALAAttention::reset_cache() {
    k_cache_ = infinicore::Tensor();
    v_cache_ = infinicore::Tensor();
    kv_capacity_ = 0;
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
    (void)kv_cache;
    (void)input_offsets;
    (void)cu_seqlens;
    (void)block_tables;
    (void)slot_mapping;
    return forward_dense_(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths);
}

infinicore::Tensor MiniCPMSALAAttention::forward_dense_(const infinicore::Tensor &hidden_states,
                                                       const infinicore::Tensor &position_ids,
                                                       std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                                       std::optional<infinicore::Tensor> past_sequence_lengths,
                                                       std::optional<infinicore::Tensor> total_sequence_lengths) const {
    // Input: [B, S, H]
    auto shape = hidden_states->shape();
    const size_t batch_size = shape[0];
    const size_t seq_len = shape[1];

    // #region agent log
    if (layer_idx_ == 0) {
        const char *log_path = std::getenv("INFINI_DEBUG_LOG");
        if (log_path) {
            std::ofstream log(log_path, std::ios::app);
            if (log) {
                const auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::system_clock::now().time_since_epoch())
                                        .count();
                log << "{\"sessionId\":\"9146ea\",\"hypothesisId\":\"H3\",\"location\":\"minicpm_sala_attention.cpp:forward_dense_entry\",\"message\":\"Inf layer0 attn env/config\",\"data\":{"
                    << "\"INFINI_DEBUG_ATTN_DUMP\":" << (std::getenv("INFINI_DEBUG_ATTN_DUMP") ? 1 : 0) << ","
                    << "\"use_rope\":" << (use_rope_ ? 1 : 0) << ","
                    << "\"use_qk_norm\":" << (use_qk_norm_ ? 1 : 0) << ","
                    << "\"use_output_gate\":" << (use_output_gate_ ? 1 : 0) << ","
                    << "\"use_output_norm\":" << (use_output_norm_ ? 1 : 0) << ","
                    << "\"is_sparse_layer\":" << (is_sparse_layer_ ? 1 : 0) << ","
                    << "\"n_head\":" << num_attention_heads_ << ","
                    << "\"n_kv\":" << num_key_value_heads_ << ","
                    << "\"head_dim\":" << head_dim_ << ","
                    << "\"scaling\":" << scaling_ << ","
                    << "\"batch\":" << batch_size << ","
                    << "\"seqlen\":" << seq_len
                    << "},\"timestamp\":" << now_ms << "}\n";
            }
        }
    }
    // #endregion

    auto hs_mut = hidden_states;
    auto q = q_proj_->forward(hs_mut);
    auto k = k_proj_->forward(hs_mut);
    auto v = v_proj_->forward(hs_mut);
    q = q->contiguous();
    k = k->contiguous();
    v = v->contiguous();

    // Reshape: [B, S, n_head, head_dim]
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

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
    if (total_sequence_lengths.has_value()) {
        total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
    }

    auto q_perm = q_reshaped->permute({0, 2, 1, 3})->contiguous(); // [B, n_q, S, D]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3})->contiguous(); // [B, n_kv, S, D]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3})->contiguous(); // [B, n_kv, S, D]

    // Dense fallback KV cache (per-layer).
    // The engine passes only the newly generated token on decode steps (seq_len=1),
    // so we must accumulate K/V to attend over the full prefix.
    infinicore::Tensor k_total = k_permuted;
    infinicore::Tensor v_total = v_permuted;
    if (past_sequence_lengths.has_value() && total_sequence_lengths.has_value()) {
        size_t cache_pos = reinterpret_cast<int32_t *>(
            past_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
        const size_t update_len = seq_len;
        const size_t max_cache_len = model_config_->get<size_t>("max_position_embeddings");

        // Grow-on-demand to avoid allocating huge max_position_embeddings upfront.
        const size_t needed_len = std::max(total_seq_len, cache_pos + update_len);
        if (needed_len > max_cache_len) {
            throw std::runtime_error("MiniCPMSALAAttention: needed KV length exceeds max_position_embeddings");
        }

        if (kv_capacity_ < needed_len) {
            size_t new_cap = (kv_capacity_ == 0) ? 128 : kv_capacity_;
            while (new_cap < needed_len) new_cap *= 2;
            new_cap = std::min(new_cap, max_cache_len);

            auto new_k = infinicore::Tensor::empty({batch_size, num_key_value_heads_, new_cap, head_dim_},
                                                   k_permuted->dtype(), k_permuted->device());
            auto new_v = infinicore::Tensor::empty({batch_size, num_key_value_heads_, new_cap, head_dim_},
                                                   v_permuted->dtype(), v_permuted->device());

            if (k_cache_ && v_cache_) {
                auto k_dst = new_k->narrow({{2, 0, kv_capacity_}});
                auto v_dst = new_v->narrow({{2, 0, kv_capacity_}});
                k_dst->copy_from(k_cache_->narrow({{2, 0, kv_capacity_}}));
                v_dst->copy_from(v_cache_->narrow({{2, 0, kv_capacity_}}));
            }

            k_cache_ = new_k;
            v_cache_ = new_v;
            kv_capacity_ = new_cap;
        }

        auto k_cache_update = k_cache_->narrow({{2, cache_pos, update_len}});
        auto v_cache_update = v_cache_->narrow({{2, cache_pos, update_len}});
        k_cache_update->copy_from(k_permuted);
        v_cache_update->copy_from(v_permuted);

        k_total = k_cache_;
        v_total = v_cache_;
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

    infinicore::Tensor attn_output;
    if (!is_sparse_layer_) {
        // Lightning / dense layers: use GLA-style grouped attention op.
        auto q_for_gla = q_perm->view({batch_size, num_attention_heads_, seq_len, head_dim_});
        auto k_for_gla = k_total;
        auto v_for_gla = v_total;
        auto gla_out = infinicore::op::gla_attention(q_for_gla, k_for_gla, v_for_gla, scaling_, true);
        attn_output = gla_out->permute({0, 2, 1, 3})
                           ->contiguous()
                           ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
    } else {
        // Sparse (minicpm4) layers:
        // Prefer InfLLM-V2 varlen kernels when available. If InfLLM-V2 is not
        // enabled in InfiniCore, optionally use the FlashAttention-style
        // mha_varlen path, and only then fall back to dense matmul.
#if defined(ENABLE_INFLLMV2)
        // Ensure contiguity before flattening views, especially after narrow() on caches.
        auto q_contig = q_perm->contiguous();
        auto k_contig = k_total->contiguous();
        auto v_contig = v_total->contiguous();

        const size_t total_q = batch_size * seq_len;
        const size_t total_k = batch_size * total_seq_len;

        auto q_varlen = q_contig->view({total_q, num_attention_heads_, head_dim_});
        auto k_varlen = k_contig->view({total_k, num_key_value_heads_, head_dim_});
        auto v_varlen = v_contig->view({total_k, num_key_value_heads_, head_dim_});

        // Single-request cu_seqlens: [0, total_q] / [0, total_k]
        auto cu_q = infinicore::Tensor::empty({2}, infinicore::DataType::I32, q_varlen->device());
        auto cu_k = infinicore::Tensor::empty({2}, infinicore::DataType::I32, k_varlen->device());
        {
            auto cu_q_cpu = cu_q->to(infinicore::Device::cpu());
            auto cu_k_cpu = cu_k->to(infinicore::Device::cpu());
            auto *q_ptr = reinterpret_cast<int32_t *>(cu_q_cpu->data());
            auto *k_ptr = reinterpret_cast<int32_t *>(cu_k_cpu->data());
            q_ptr[0] = 0;
            q_ptr[1] = static_cast<int32_t>(total_q);
            k_ptr[0] = 0;
            k_ptr[1] = static_cast<int32_t>(total_k);
            cu_q->copy_from(cu_q_cpu);
            cu_k->copy_from(cu_k_cpu);
        }

        auto out_varlen = infinicore::op::infllmv2_varlen(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_q,
            cu_k,
            static_cast<int>(seq_len),
            static_cast<int>(total_seq_len),
            scaling_,
            /*causal=*/true);

        auto out_view = out_varlen->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        attn_output = out_view->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
#elif defined(ENABLE_FLASH_ATTN)
        // Ensure contiguity before flattening views, especially after narrow() on caches.
        auto q_contig = q_perm->contiguous();
        auto k_contig = k_total->contiguous();
        auto v_contig = v_total->contiguous();

        const size_t total_q = batch_size * seq_len;
        const size_t total_k = batch_size * total_seq_len;

        auto q_varlen = q_contig->view({total_q, num_attention_heads_, head_dim_});
        auto k_varlen = k_contig->view({total_k, num_key_value_heads_, head_dim_});
        auto v_varlen = v_contig->view({total_k, num_key_value_heads_, head_dim_});

        // Single-request cu_seqlens: [0, total_q] / [0, total_k]
        auto cu_q = infinicore::Tensor::empty({2}, infinicore::DataType::I32, q_varlen->device());
        auto cu_k = infinicore::Tensor::empty({2}, infinicore::DataType::I32, k_varlen->device());
        {
            auto cu_q_cpu = cu_q->to(infinicore::Device::cpu());
            auto cu_k_cpu = cu_k->to(infinicore::Device::cpu());
            auto *q_ptr = reinterpret_cast<int32_t *>(cu_q_cpu->data());
            auto *k_ptr = reinterpret_cast<int32_t *>(cu_k_cpu->data());
            q_ptr[0] = 0;
            q_ptr[1] = static_cast<int32_t>(total_q);
            k_ptr[0] = 0;
            k_ptr[1] = static_cast<int32_t>(total_k);
            cu_q->copy_from(cu_q_cpu);
            cu_k->copy_from(cu_k_cpu);
        }

        auto dummy_block_table = infinicore::Tensor::zeros({1, 1}, cu_q->dtype(), cu_q->device());
        auto out_varlen = infinicore::op::mha_varlen(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_q,
            cu_k,
            dummy_block_table,
            static_cast<int>(seq_len),
            static_cast<int>(total_seq_len),
            std::nullopt,
            scaling_);

        auto out_view = out_varlen->view({batch_size, seq_len, num_attention_heads_, head_dim_});
        attn_output = out_view->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
#else
        const size_t ngroup = num_attention_heads_ / num_key_value_heads_;
        auto Q = q_perm->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
        auto K = k_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});

        auto Kt = K->permute({0, 2, 1})->contiguous();
        auto attn_weight = infinicore::op::matmul(Q, Kt, scaling_);

        auto attn_weight_softmax = attn_weight->view({batch_size * num_attention_heads_, seq_len, total_seq_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

        auto out = infinicore::op::matmul(attn_weight, V); // [B*n_kv, ng*S, D]
        attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                           ->permute({0, 2, 1, 3})
                           ->contiguous()
                           ->view({batch_size, seq_len, num_attention_heads_ * head_dim_});
#endif
    }

    // #region agent log
    if (layer_idx_ < 2 && std::getenv("INFINI_DEBUG_ATTN_DUMP")) {
        const char *log_path = std::getenv("INFINI_DEBUG_LOG");
        if (log_path) {
            auto dump_stats = [&](const infinicore::Tensor &t,
                                  const char *msg,
                                  const char *loc,
                                  const char *bin_path) {
                auto cpu_t = t->to(infinicore::Device::cpu());
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
                if (bin_path) {
                    std::ofstream bin(bin_path, std::ios::binary);
                    if (bin) bin.write(reinterpret_cast<const char *>(f32_buf.data()), n * sizeof(float));
                }
                float mn = f32_buf.empty() ? 0 : f32_buf[0], mx = mn, sum = 0, ss = 0;
                for (float v : f32_buf) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ss += v * v; }
                const double norm = ss > 0 ? std::sqrt(ss) : 0.0;
                std::ofstream log(log_path, std::ios::app);
                if (log) {
                    std::string shape_json = "[";
                    for (size_t i = 0; i < shp.size(); ++i) shape_json += (i ? "," : "") + std::to_string(shp[i]);
                    shape_json += "]";
                    log << "{\"sessionId\":\"9146ea\",\"hypothesisId\":\"H1\",\"location\":\"" << loc << "\",\"message\":\"" << msg << "\",\"data\":{\"scaling_\":" << scaling_ << ",\"head_dim\":" << head_dim_ << ",\"shape\":" << shape_json << ",\"min\":" << mn << ",\"max\":" << mx << ",\"mean\":" << (n ? sum / n : 0) << ",\"l2\":" << norm << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "}\n";
                }
            };

            // Pre-gate attention output (matches earlier behavior for layer0; extended to layer1).
            const char *loc = (layer_idx_ == 0)
                                  ? "minicpm_sala_attention.cpp:attn_pre_gate_l0"
                                  : "minicpm_sala_attention.cpp:attn_pre_gate_l1";
            const char *msg = (layer_idx_ == 0)
                                  ? "Inf layer0 attn pre-gate"
                                  : "Inf layer1 attn pre-gate";
            const char *bin = (layer_idx_ == 0) ? "/tmp/inf_attn_out_layer0.bin" : nullptr;
            dump_stats(attn_output, msg, loc, bin);
        }
    }
    // #endregion

    // Output norm + gate variants
    if (use_output_gate_) {
        if (o_gate_) {
            // Sparse (minicpm4): y = sigmoid(o_gate(x)) * attn_output
            auto gate_in = hidden_states;
            auto gate = o_gate_->forward(gate_in);
            infinicore::op::sigmoid_(gate, gate);
            attn_output = infinicore::op::mul(attn_output, gate);
        } else if (z_proj_) {
            // Lightning: y = (attn_output_normed) * silu(z_proj(x)) (approx)
            // In SALA code, this is an output gating; we implement with SiLU(z) as gate.
            auto z_in = hidden_states;
            auto z = z_proj_->forward(z_in);
            infinicore::op::silu_(z, z);
            // Optional per-head output norm (o_norm) on [B,S,n_head,head_dim]
            if (use_output_norm_ && o_norm_) {
                // o_norm is defined over hidden_size.
                attn_output = o_norm_->forward(attn_output);
            }
            attn_output = infinicore::op::mul(attn_output, z);
        }
    } else if (use_output_norm_ && o_norm_) {
        attn_output = o_norm_->forward(attn_output);
    }

    auto attn_out_mut = attn_output;
    auto out = o_proj_->forward(attn_out_mut);

    // #region agent log
    if (layer_idx_ < 2 && std::getenv("INFINI_DEBUG_ATTN_DUMP")) {
        const char *log_path = std::getenv("INFINI_DEBUG_LOG");
        if (log_path) {
            // Mirror the same stats helper (minimal duplication)
            auto dump_stats = [&](const infinicore::Tensor &t,
                                  const char *msg,
                                  const char *loc,
                                  const char *bin_path) {
                auto cpu_t = t->to(infinicore::Device::cpu());
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
                if (bin_path) {
                    std::ofstream bin(bin_path, std::ios::binary);
                    if (bin) bin.write(reinterpret_cast<const char *>(f32_buf.data()), n * sizeof(float));
                }
                float mn = f32_buf.empty() ? 0 : f32_buf[0], mx = mn, sum = 0, ss = 0;
                for (float v : f32_buf) { mn = std::min(mn, v); mx = std::max(mx, v); sum += v; ss += v * v; }
                const double norm = ss > 0 ? std::sqrt(ss) : 0.0;
                std::ofstream log(log_path, std::ios::app);
                if (log) {
                    std::string shape_json = "[";
                    for (size_t i = 0; i < shp.size(); ++i) shape_json += (i ? "," : "") + std::to_string(shp[i]);
                    shape_json += "]";
                    log << "{\"sessionId\":\"9146ea\",\"hypothesisId\":\"H2\",\"location\":\"" << loc << "\",\"message\":\"" << msg << "\",\"data\":{\"shape\":" << shape_json << ",\"min\":" << mn << ",\"max\":" << mx << ",\"mean\":" << (n ? sum / n : 0) << ",\"l2\":" << norm << "},\"timestamp\":" << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() << "}\n";
                }
            };
            const char *loc_gate = (layer_idx_ == 0)
                                       ? "minicpm_sala_attention.cpp:attn_post_gate_l0"
                                       : "minicpm_sala_attention.cpp:attn_post_gate_l1";
            const char *loc_oproj = (layer_idx_ == 0)
                                         ? "minicpm_sala_attention.cpp:attn_post_oproj_l0"
                                         : "minicpm_sala_attention.cpp:attn_post_oproj_l1";
            const char *msg_gate = (layer_idx_ == 0)
                                       ? "Inf layer0 attn post-gate/norm"
                                       : "Inf layer1 attn post-gate/norm";
            const char *msg_oproj = (layer_idx_ == 0)
                                        ? "Inf layer0 attn post-o_proj"
                                        : "Inf layer1 attn post-o_proj";
            dump_stats(attn_output, msg_gate, loc_gate, nullptr);
            dump_stats(out, msg_oproj, loc_oproj, nullptr);
        }
    }
    // #endregion

    return out;
}

} // namespace infinilm::models::minicpm_sala
