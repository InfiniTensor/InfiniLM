#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cstdlib>

void createDeviceResource(JiugeDeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_q_norm, w_attn_k_norm, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }

        if (weights->attn_q_norm != nullptr) {
            w_attn_q_norm.push_back(
                getAttnQNorm(meta, weights, layer));
            w_attn_k_norm.push_back(
                getAttnKNorm(meta, weights, layer));
        }
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
        w_ffn_gate_up.push_back(
            getFFNGateUp(meta, weights, layer, idev, ndev));
        w_ffn_down.push_back(
            getFFNDown(meta, weights, layer, idev, ndev));
    }
    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = JiugeDeviceResource{
        device,
        dev_id,
        handle,
        getInEmbd(meta, weights),
        getOutNorm(meta, weights),
        getOutEmbd(meta, weights),
        getSinTable(meta),
        getCosTable(meta),
        w_attn_norm,
        w_attn_qkv,
        b_attn_qkv,
        w_attn_q_norm,
        w_attn_k_norm,
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate_up,
        w_ffn_down,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(JiugeDeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    for (auto &t : res.w_attn_norm) {
        t.reset();
    }
    res.w_attn_norm.clear();
    for (auto &t : res.w_attn_qkv) {
        t.reset();
    }
    res.w_attn_qkv.clear();
    for (auto &t : res.b_attn_qkv) {
        t.reset();
    }
    res.b_attn_qkv.clear();
    for (auto &t : res.w_attn_out) {
        t.reset();
    }
    res.w_attn_out.clear();
    for (auto &t : res.w_ffn_norm) {
        t.reset();
    }
    res.w_ffn_norm.clear();
    for (auto &t : res.w_ffn_gate_up) {
        t.reset();
    }
    res.w_ffn_gate_up.clear();
    for (auto &t : res.w_ffn_down) {
        t.reset();
    }
    res.w_ffn_down.clear();
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferDeviceBatch(const JiugeMeta &meta, JiugeDeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      const float *repetition_penalty,
                      const uint32_t *full_tokens, uint32_t full_ntok,
                      uint32_t *output, void *last_logits) {
    // #region agent log - Function entry
    static bool debug_func_entry = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
    if (debug_func_entry && idev == 0 && nreq > 0 && req_pos[0] == 0) {
        std::cerr << "[FUNC_ENTRY] inferDeviceBatch called" << std::endl;
    }
    // #endregion agent log
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;
    bool has_qk_norm = rsrc.w_attn_q_norm.size() > 0 && rsrc.w_attn_k_norm.size() > 0;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_rope->slice(1, 0, nh);
    auto k_buf = qkv_rope->slice(1, nh, nkvh);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // MLP buffers
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);
        // qkv_proj
        linear(qkv_buf, logits_out, rsrc.w_attn_qkv[layer], 1.0, 0.0, nullptr, has_qkv_bias ? rsrc.b_attn_qkv[layer] : nullptr);

        if (has_qk_norm) {
            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon);
            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon);
        }

        // rope
        rope(q_buf, q_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        rope(k_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
            auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // o_proj
        linear(logits_in, o_buf, rsrc.w_attn_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta.epsilon);
        linear(gate_up_buf, logits_out, rsrc.w_ffn_gate_up[layer], 1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf, rsrc.w_ffn_down[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }
    // Sample and Output
    if (idev == 0) {
        // Calculate output scaling factor: dim_model_base / hidden_size
        // This matches PyTorch: logits = lm_head(hidden_states / (hidden_size / dim_model_base))
        float output_scale = meta.dim_model_base > 0 ? (float)meta.dim_model_base / (float)meta.d : 1.0f;

        // #region agent log
        static bool debug_scaling = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
        if (debug_scaling && nreq > 0 && req_pos[0] == 0) {
            std::cerr << "[SCALING_DEBUG] dim_model_base=" << meta.dim_model_base
                      << " hidden_size=" << meta.d
                      << " output_scale=" << output_scale << std::endl;
        }
        // #endregion agent log

        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, rsrc.w_out_norm, meta.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, rsrc.w_out_embd, output_scale, 0.0, nullptr, nullptr);

            auto log_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            logSoftmax(log_logits_buf, last_logits_buf);

            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, log_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        rsrc.w_out_norm,
                        meta.epsilon);
            }

            // #region agent log - Investigate hidden state and weight magnitudes
            static bool debug_investigate = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
            if (debug_investigate && nreq > 0 && (req_pos[0] == 0 || (req_pos[0] >= 1345 && req_pos[0] < 1380))) {
                RUN_INFINI(infinirtStreamSynchronize(stream));
                size_t last_token_idx = token_offset - 1;
                auto sample_size = std::min(100UL, d);
                auto hidden_before = std::vector<float>(sample_size);
                auto hidden_after = std::vector<float>(sample_size);
                auto weight_sample = std::vector<float>(sample_size);

                // Sample hidden states before RMSNorm
                if (dt_logits == INFINI_DTYPE_F32) {
                    RUN_INFINI(infinirtMemcpy(hidden_before.data(), logits_in->data(last_token_idx * d), sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
                    RUN_INFINI(infinirtMemcpy(hidden_after.data(), logits_out->slice(0, 0, 1)->data(), sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
                } else if (dt_logits == INFINI_DTYPE_F16) {
                    auto raw_before = std::vector<uint16_t>(sample_size);
                    auto raw_after = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw_before.data(), logits_in->data(last_token_idx * d), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    RUN_INFINI(infinirtMemcpy(raw_after.data(), logits_out->slice(0, 0, 1)->data(), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) {
                        hidden_before[i] = f16_to_f32(raw_before[i]);
                        hidden_after[i] = f16_to_f32(raw_after[i]);
                    }
                } else if (dt_logits == INFINI_DTYPE_BF16) {
                    auto raw_before = std::vector<uint16_t>(sample_size);
                    auto raw_after = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw_before.data(), logits_in->data(last_token_idx * d), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    RUN_INFINI(infinirtMemcpy(raw_after.data(), logits_out->slice(0, 0, 1)->data(), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) {
                        hidden_before[i] = bf16_to_f32(raw_before[i]);
                        hidden_after[i] = bf16_to_f32(raw_after[i]);
                    }
                }

                // Sample output embedding weights
                auto weight_dtype = rsrc.w_out_embd->dtype();
                if (weight_dtype == INFINI_DTYPE_F32) {
                    RUN_INFINI(infinirtMemcpy(weight_sample.data(), rsrc.w_out_embd->data(0), sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
                } else if (weight_dtype == INFINI_DTYPE_F16) {
                    auto raw = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw.data(), rsrc.w_out_embd->data(0), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) weight_sample[i] = f16_to_f32(raw[i]);
                } else if (weight_dtype == INFINI_DTYPE_BF16) {
                    auto raw = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw.data(), rsrc.w_out_embd->data(0), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) weight_sample[i] = bf16_to_f32(raw[i]);
                }

                float max_hb = *std::max_element(hidden_before.begin(), hidden_before.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
                float max_ha = *std::max_element(hidden_after.begin(), hidden_after.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
                float max_w = *std::max_element(weight_sample.begin(), weight_sample.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });

                std::cerr << "[INVESTIGATE] req_pos=" << req_pos[0]
                          << " hidden_before_rmsnorm_max_abs=" << std::abs(max_hb)
                          << " hidden_after_rmsnorm_max_abs=" << std::abs(max_ha)
                          << " output_embd_weight_max_abs=" << std::abs(max_w)
                          << " output_scale=" << output_scale << std::endl;
            }
            // #endregion agent log

            // #region agent log - Keep output_scale to compensate for removing scale_output from norm weights
            // We removed scale_output from norm weights, so we need to apply scaling in linear projection
            // This matches PyTorch: logits = lm_head(hidden_states / (hidden_size / dim_model_base))
            // #endregion agent log
            linear(prob_buf, logits_out->slice(0, 0, nreq), rsrc.w_out_embd, output_scale, 0.0, nullptr, nullptr);

            // Apply repetition penalty if needed
            if (repetition_penalty != nullptr) {
                // Check if any penalty != 1.0
                bool need_penalty = false;
                for (uint32_t req = 0; req < nreq; req++) {
                    if (repetition_penalty[req] != 1.0f) {
                        need_penalty = true;
                        break;
                    }
                }

                if (need_penalty) {
                    // Create boolean mask [nreq, dvoc] on host
                    // Use uint8_t instead of bool because std::vector<bool> doesn't have data() method
                    std::vector<uint8_t> mask_host(nreq * dvoc, 0);

                    // Extract previous tokens and set mask
                    // tokens array contains only NEW tokens (for embedding lookup - keep separate!)
                    // full_tokens array contains full token history (for repetition penalty mask only)
                    if (full_tokens != nullptr && full_ntok > 0) {
                        // Use full token history for repetition penalty mask
                        size_t req_token_start = 0;
                        for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
                            // Calculate where this request's tokens start in the full tokens array
                            // For single request (nreq=1), req_token_start should always be 0
                            // For multiple requests, accumulate the lengths of previous requests
                            if (req_idx > 0) {
                                req_token_start += req_pos[req_idx - 1] + req_lens[req_idx - 1];
                            }

                            // Mark all tokens from 0 to req_pos[req_idx] + req_lens[req_idx] - 1 (excluding last)
                            // total_seq_len = current position + new tokens in this batch
                            uint32_t total_seq_len = req_pos[req_idx] + req_lens[req_idx];
                            // actual_len = number of tokens to mark (all previous tokens, excluding the one being predicted)
                            uint32_t actual_len = total_seq_len - 1;

                            // Ensure we don't exceed the full_tokens array bounds
                            if (req_token_start + actual_len > full_ntok) {
                                actual_len = (req_token_start < full_ntok) ? (full_ntok - req_token_start) : 0;
                            }

                            // #region agent log - Trace mask creation (dynamic token tracking)
                            static bool debug_mask = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
                            bool should_trace = debug_mask && req_idx == 0 && (req_pos[req_idx] >= 1345 && req_pos[req_idx] < 1380);
                            // #endregion agent log

                            // Mark tokens in mask
                            for (uint32_t i = 0; i < actual_len; i++) {
                                if (req_token_start + i < full_ntok) {
                                    uint32_t token_id = full_tokens[req_token_start + i];
                                    if (token_id < dvoc) {
                                        mask_host[req_idx * dvoc + token_id] = 1;
                                    }
                                }
                            }

                            // #region agent log - Trace mask creation details
                            if (should_trace) {
                                std::cerr << "[MASK_TRACE] req_pos=" << req_pos[req_idx]
                                          << " req_lens=" << req_lens[req_idx]
                                          << " req_token_start=" << req_token_start
                                          << " total_seq_len=" << total_seq_len
                                          << " actual_len=" << actual_len
                                          << " full_ntok=" << full_ntok << std::endl;
                                // Count tokens in full_tokens range
                                std::map<uint32_t, uint32_t> token_counts;
                                for (uint32_t i = 0; i < actual_len && req_token_start + i < full_ntok; i++) {
                                    uint32_t token_id = full_tokens[req_token_start + i];
                                    if (token_id < dvoc) {
                                        token_counts[token_id]++;
                                    }
                                }
                                // Find most frequent token
                                uint32_t most_freq_token = 0;
                                uint32_t most_freq_count = 0;
                                for (const auto& pair : token_counts) {
                                    if (pair.second > most_freq_count) {
                                        most_freq_count = pair.second;
                                        most_freq_token = pair.first;
                                    }
                                }
                                std::cerr << "[MASK_TRACE] Most frequent token in range: " << most_freq_token
                                          << " (appears " << most_freq_count << " times)" << std::endl;
                                std::cerr << "[MASK_TRACE] mask_host[" << most_freq_token << "] = "
                                          << (int)mask_host[req_idx * dvoc + most_freq_token] << std::endl;
                                // Check last few tokens in full_tokens
                                uint32_t check_start = (req_token_start + actual_len > 10) ? (req_token_start + actual_len - 10) : req_token_start;
                                std::cerr << "[MASK_TRACE] Last 10 tokens in range: ";
                                for (uint32_t i = check_start; i < req_token_start + actual_len && i < full_ntok; i++) {
                                    std::cerr << full_tokens[i] << " ";
                                }
                                std::cerr << std::endl;
                            }
                            // #endregion agent log
                        }
                    } else {
                        // Fallback: only penalize tokens in current batch if full_tokens not available
                        size_t token_offset = 0;
                        for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
                            if (req_lens[req_idx] > 1) {
                                for (uint32_t i = 0; i < req_lens[req_idx] - 1; i++) {
                                    uint32_t token_id = tokens[token_offset + i];
                                    if (token_id < dvoc) {
                                        mask_host[req_idx * dvoc + token_id] = 1;
                                    }
                                }
                            }
                            token_offset += req_lens[req_idx];
                        }
                    }

                    // Create mask tensor and copy from host
                    auto mask_buf = Tensor::buffer(INFINI_DTYPE_BOOL, {nreq, dvoc}, rsrc.memory_pool);
                    RUN_INFINI(infinirtMemcpyAsync(mask_buf->data(), mask_host.data(),
                                                  nreq * dvoc * sizeof(uint8_t),
                                                  INFINIRT_MEMCPY_H2D, stream));

                    // Allocate device buffer for penalties
                    auto d_penalties = Tensor::buffer(INFINI_DTYPE_F32, {nreq}, rsrc.memory_pool);
                    RUN_INFINI(infinirtMemcpyAsync(d_penalties->data(), repetition_penalty,
                                                  nreq * sizeof(float),
                                                  INFINIRT_MEMCPY_H2D, stream));

                    // Debug logging (enable with REPETITION_PENALTY_DEBUG=1)
                    static bool debug_enabled = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
                    bool should_log = false;
                    if (debug_enabled && nreq > 0) {
                        // Log at regular intervals to track logit magnitude over time
                        should_log = (req_pos[0] % 50 == 0) || (req_pos[0] >= 1345 && req_pos[0] < 1380);
                    }

                    // Sample logits before penalty to track magnitude
                    std::vector<float> logits_before;
                    std::vector<std::pair<float, uint32_t>> logit_pairs_before;  // Save for after-penalty comparison
                    uint32_t repeating_tokens[] = {18867, 6686, 6783, 59466, 59566};
                    if (should_log) {
                        RUN_INFINI(infinirtStreamSynchronize(stream));

                        // Read logits and convert to float based on dtype
                        auto prob_host = std::vector<float>(nreq * dvoc);
                        if (dt_logits == INFINI_DTYPE_F32) {
                            RUN_INFINI(infinirtMemcpy(prob_host.data(), prob_buf->data(),
                                                      nreq * dvoc * sizeof(float),
                                                      INFINIRT_MEMCPY_D2H));
                        } else {
                            auto raw_buf = std::vector<uint8_t>(nreq * dvoc * dsize(dt_logits));
                            RUN_INFINI(infinirtMemcpy(raw_buf.data(), prob_buf->data(),
                                                      nreq * dvoc * dsize(dt_logits),
                                                      INFINIRT_MEMCPY_D2H));
                            if (dt_logits == INFINI_DTYPE_F16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t h = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = f16_to_f32(h);
                                }
                            } else if (dt_logits == INFINI_DTYPE_BF16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t b = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = bf16_to_f32(b);
                                }
                            }
                        }

                        // Calculate logit statistics
                        float max_logit = prob_host[0];
                        float min_logit = prob_host[0];
                        float sum_logit = 0.0f;
                        for (size_t i = 0; i < nreq * dvoc; i++) {
                            if (prob_host[i] > max_logit) max_logit = prob_host[i];
                            if (prob_host[i] < min_logit) min_logit = prob_host[i];
                            sum_logit += prob_host[i];
                        }
                        float avg_logit = sum_logit / (nreq * dvoc);

                        // Find top 5 logits
                        logit_pairs_before.clear();
                        for (uint32_t i = 0; i < dvoc; i++) {
                            logit_pairs_before.push_back({prob_host[0 * dvoc + i], i});
                        }
                        std::sort(logit_pairs_before.begin(), logit_pairs_before.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });

                        for (uint32_t tidx = 0; tidx < 5; tidx++) {
                            uint32_t token_id = repeating_tokens[tidx];
                            if (token_id < dvoc) {
                                logits_before.push_back(prob_host[0 * dvoc + token_id]);
                            }
                        }

                        uint32_t mask_ones = 0;
                        for (uint32_t i = 0; i < nreq * dvoc; i++) {
                            if (mask_host[i] != 0) mask_ones++;
                        }

                        std::cerr << "[REP_PENALTY] req_pos=" << req_pos[0]
                                  << " logits: max=" << max_logit << " min=" << min_logit
                                  << " avg=" << avg_logit
                                  << " mask=" << mask_ones << " marked"
                                  << " penalty=" << repetition_penalty[0] << std::endl;

                        // Log top 5 logits to see which tokens are dominant
                        if (req_pos[0] >= 1345 && req_pos[0] < 1380) {
                            std::cerr << "  Top 5 logits BEFORE penalty: ";
                            for (uint32_t i = 0; i < 5 && i < logit_pairs_before.size(); i++) {
                                std::cerr << "token[" << logit_pairs_before[i].second << "]=" << logit_pairs_before[i].first << " ";
                            }
                            std::cerr << std::endl;
                        }
                    }

                    // #region agent log - Verify mask before applying penalty (dynamic top token tracking)
                    static bool debug_mask_before = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
                    if (debug_mask_before && nreq > 0 && (req_pos[0] >= 1345 && req_pos[0] < 1380)) {
                        RUN_INFINI(infinirtStreamSynchronize(stream));
                        // Read logits to find top token
                        auto prob_host = std::vector<float>(nreq * dvoc);
                        if (dt_logits == INFINI_DTYPE_F32) {
                            RUN_INFINI(infinirtMemcpy(prob_host.data(), prob_buf->data(),
                                                      nreq * dvoc * sizeof(float),
                                                      INFINIRT_MEMCPY_D2H));
                        } else {
                            auto raw_buf = std::vector<uint8_t>(nreq * dvoc * dsize(dt_logits));
                            RUN_INFINI(infinirtMemcpy(raw_buf.data(), prob_buf->data(),
                                                      nreq * dvoc * dsize(dt_logits),
                                                      INFINIRT_MEMCPY_D2H));
                            if (dt_logits == INFINI_DTYPE_F16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t h = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = f16_to_f32(h);
                                }
                            } else if (dt_logits == INFINI_DTYPE_BF16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t b = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = bf16_to_f32(b);
                                }
                            }
                        }
                        // Find top token
                        uint32_t top_token = 0;
                        float max_logit = prob_host[0];
                        for (uint32_t i = 1; i < dvoc; i++) {
                            if (prob_host[i] > max_logit) {
                                max_logit = prob_host[i];
                                top_token = i;
                            }
                        }
                        // Copy mask back from device to verify
                        std::vector<uint8_t> mask_check(nreq * dvoc);
                        RUN_INFINI(infinirtMemcpy(mask_check.data(), mask_buf->data(),
                                                  nreq * dvoc * sizeof(uint8_t),
                                                  INFINIRT_MEMCPY_D2H));
                        std::cerr << "[MASK_VERIFY] req_pos=" << req_pos[0]
                                  << " top_token=" << top_token
                                  << " top_logit=" << max_logit
                                  << " mask[" << top_token << "]=" << (int)mask_check[top_token] << std::endl;
                    }
                    // #endregion agent log

                    // Apply repetition penalty
                    applyRepetitionPenalty(prob_buf, mask_buf,
                                          static_cast<const float *>(d_penalties->data()));

                    // Sample logits after penalty
                    if (should_log) {
                        RUN_INFINI(infinirtStreamSynchronize(stream));

                        auto prob_host = std::vector<float>(nreq * dvoc);
                        if (dt_logits == INFINI_DTYPE_F32) {
                            RUN_INFINI(infinirtMemcpy(prob_host.data(), prob_buf->data(),
                                                      nreq * dvoc * sizeof(float),
                                                      INFINIRT_MEMCPY_D2H));
                        } else {
                            auto raw_buf = std::vector<uint8_t>(nreq * dvoc * dsize(dt_logits));
                            RUN_INFINI(infinirtMemcpy(raw_buf.data(), prob_buf->data(),
                                                      nreq * dvoc * dsize(dt_logits),
                                                      INFINIRT_MEMCPY_D2H));
                            if (dt_logits == INFINI_DTYPE_F16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t h = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = f16_to_f32(h);
                                }
                            } else if (dt_logits == INFINI_DTYPE_BF16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t b = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = bf16_to_f32(b);
                                }
                            }
                        }

                        // #region agent log - Log top 5 tokens before/after penalty
                        if (req_pos[0] >= 1345 && req_pos[0] < 1380) {
                            // Find top 5 tokens after penalty
                            std::vector<std::pair<float, uint32_t>> logit_pairs_after;
                            for (uint32_t i = 0; i < dvoc; i++) {
                                logit_pairs_after.push_back({prob_host[0 * dvoc + i], i});
                            }
                            std::sort(logit_pairs_after.begin(), logit_pairs_after.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });

                            std::cerr << "[REP_PENALTY_AFTER] req_pos=" << req_pos[0] << " Top 5 tokens AFTER penalty:" << std::endl;
                            for (uint32_t i = 0; i < 5 && i < logit_pairs_after.size(); i++) {
                                uint32_t token_id = logit_pairs_after[i].second;
                                float logit_after = logit_pairs_after[i].first;
                                // Find corresponding before value from logit_pairs
                                float logit_before = 0.0f;
                                for (const auto& pair : logit_pairs_before) {
                                    if (pair.second == token_id) {
                                        logit_before = pair.first;
                                        break;
                                    }
                                }
                                bool is_marked = (mask_host[0 * dvoc + token_id] != 0);
                                float change = logit_after - logit_before;
                                std::cerr << "  token[" << token_id << "]: " << logit_before
                                          << " -> " << logit_after << " (Δ" << change << ") "
                                          << (is_marked ? "[MARKED]" : "[NOT MARKED]") << std::endl;
                            }
                        }
                        // #endregion agent log

                        std::cerr << "[REP_PENALTY] Operator called. Logit changes:" << std::endl;
                        for (uint32_t tidx = 0; tidx < 5; tidx++) {
                            uint32_t token_id = repeating_tokens[tidx];
                            if (token_id < dvoc && tidx < logits_before.size()) {
                                float logit_after = prob_host[0 * dvoc + token_id];
                                float logit_before = logits_before[tidx];
                                float change = logit_after - logit_before;
                                bool is_marked = (mask_host[0 * dvoc + token_id] != 0);

                                std::cerr << "  Token " << token_id
                                          << ": " << logit_before << " -> " << logit_after
                                          << " (Δ" << change << ")"
                                          << (is_marked ? " [MARKED]" : " [NOT MARKED]") << std::endl;
                            }
                        }
                    }
                }
            }

            // #region agent log - Use fixed seed for deterministic testing
            static bool use_fixed_seed = (getenv("FIXED_RANDOM_SEED") != nullptr);
            static thread_local std::mt19937 gen(use_fixed_seed ? 42 : std::random_device{}());
            // #endregion agent log
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }
}

void inferDeviceBatchPaged(const JiugeMeta &meta, JiugeDeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const int32_t *block_tables,
                      const int32_t *slot_mapping,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      const float *repetition_penalty,
                      const uint32_t *full_tokens, uint32_t full_ntok,
                      const uint32_t is_prefill, const bool enable_paged_attn,
                      uint32_t *output, void *last_logits) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto ngroup = nh / nkvh;
    // auto dctx = meta.dctx;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;
    bool has_qk_norm = rsrc.w_attn_q_norm.size() > 0 && rsrc.w_attn_k_norm.size() > 0;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    auto qkv_rope = qkv_buf->view({ntok, nh + nkvh * 2, dh});
    auto q_buf = qkv_rope->slice(1, 0, nh);
    auto k_buf = qkv_rope->slice(1, nh, nkvh);

    // Prepare inputs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    auto batch_seq_lens = std::vector<int32_t>(nreq);

    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        batch_seq_lens[req] = req_lens[req] + req_pos[req];
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    std::shared_ptr<Tensor> slot_mapping_buf, block_tables_buf, seq_lens_buf;
    size_t max_seq_len_in_batch = 0;
    if (enable_paged_attn) {
        max_seq_len_in_batch = *std::max_element(batch_seq_lens.begin(), batch_seq_lens.end());
        // Assuming block_size is a known constant, e.g., 16. The max_blocks_per_seq can be calculated.
        // Let's assume a reasonable upper bound for simplicity. This might need to be passed in.
        // TODO: get block_size from meta
        size_t block_size = meta.kvcache_block_size;
        size_t max_blocks_per_seq = (max_seq_len_in_batch + block_size - 1) / block_size;


        slot_mapping_buf = Tensor::buffer(INFINI_DTYPE_I32, {ntok}, rsrc.memory_pool);
        block_tables_buf = Tensor::buffer(INFINI_DTYPE_I32, {(uint32_t)nreq, (uint32_t)max_blocks_per_seq}, rsrc.memory_pool);
        seq_lens_buf = Tensor::buffer(INFINI_DTYPE_I32, {nreq}, rsrc.memory_pool);

        RUN_INFINI(infinirtMemcpyAsync(slot_mapping_buf->data(), slot_mapping, sizeof(int32_t) * ntok, INFINIRT_MEMCPY_H2D, stream));
        RUN_INFINI(infinirtMemcpyAsync(block_tables_buf->data(), block_tables, sizeof(int32_t) * (nreq * max_blocks_per_seq), INFINIRT_MEMCPY_H2D, stream));
        RUN_INFINI(infinirtMemcpyAsync(seq_lens_buf->data(), batch_seq_lens.data(), sizeof(int32_t) * nreq, INFINIRT_MEMCPY_H2D, stream));

    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});


    // MLP buffers
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);


    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);
        // qkv_proj
        linear(qkv_buf, logits_out, rsrc.w_attn_qkv[layer], 1.0, 0.0, nullptr, has_qkv_bias ? rsrc.b_attn_qkv[layer] : nullptr);

        if (has_qk_norm) {
            rmsnorm(q_buf, q_buf, rsrc.w_attn_q_norm[layer], meta.epsilon);
            rmsnorm(k_buf, k_buf, rsrc.w_attn_k_norm[layer], meta.epsilon);
        }

        // rope
        rope(qkv_rope->slice(1, 0, nh), qkv_rope->slice(1, 0, nh), pos_ids_buf, rsrc.sin_table, rsrc.cos_table);
        rope(qkv_rope->slice(1, nh, nkvh), qkv_rope->slice(1, nh, nkvh), pos_ids_buf, rsrc.sin_table, rsrc.cos_table);

        if (enable_paged_attn) {
            auto k = qkv_rope->slice({ {0, 0, ntok}, {1, nh, nkvh} });
            auto v = qkv_rope->slice({ {0, 0, ntok}, {1, nh + nkvh, nkvh} });

            auto k_cache_pool = kv_caches[0]->k[idev][layer];
            auto v_cache_pool = kv_caches[0]->v[idev][layer];
            pagedCaching(k, v, k_cache_pool, v_cache_pool, slot_mapping_buf);

            if (is_prefill) {
                size_t token_offset = 0;
                for (uint32_t req = 0; req < nreq; req++) {
                    auto past_len = req_pos[req];
                    auto seq_len = req_lens[req];
                    auto total_len = past_len + seq_len;
                    auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                    auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
                    auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});
                    auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                    rearrange(q_rearrange->slice(2, 0, seq_len), q);
                    auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
                    auto k_gemm = k->permute({1, 2, 0});
                    linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
                    auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
                    causalSoftmax(qk_softmax, qk_softmax);
                    auto v_gemm = v->permute({1, 0, 2});
                    linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
                    rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

                    token_offset += seq_len;
                }
            } else {
                auto o = o_buf->slice({{0, 0, ntok}})->view({ntok, nh, dh});
                auto q_batch = qkv_rope->slice({ {0, 0, ntok}, {1, 0, nh} })->view({ntok, nh, dh});
                float scale = 1.f / float(sqrt(dh));
                pagedAttention(o, q_batch, k_cache_pool, v_cache_pool,
                               block_tables_buf, seq_lens_buf, nullptr /* alibi_slopes */, scale);


            }

        } else {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto past_len = req_pos[req];
                auto seq_len = req_lens[req];
                auto total_len = past_len + seq_len;
                auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                auto q = qkv_rope->slice({{0, token_offset, seq_len}, {1, 0, nh}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                auto k = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh, nkvh}});
                auto v = qkv_rope->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}});

            // self attention
            // concat
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);
            // qk
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
            // softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

                token_offset += seq_len;
            }
        }

        // o_proj
        linear(logits_in, o_buf, rsrc.w_attn_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. FFN
        rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta.epsilon);
        linear(gate_up_buf, logits_out, rsrc.w_ffn_gate_up[layer], 1.0, 0.0, nullptr, nullptr);
        swiglu(gate_buf, up_buf, gate_buf);
        linear(logits_in, gate_buf, rsrc.w_ffn_down[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // Sample and Output
    if (idev == 0) {
        // Calculate output scaling factor: dim_model_base / hidden_size
        // This matches PyTorch: logits = lm_head(hidden_states / (hidden_size / dim_model_base))
        float output_scale = meta.dim_model_base > 0 ? (float)meta.dim_model_base / (float)meta.d : 1.0f;

        // #region agent log
        static bool debug_scaling_paged = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
        if (debug_scaling_paged && nreq > 0 && req_pos[0] == 0) {
            std::cerr << "[SCALING_DEBUG_PAGED] dim_model_base=" << meta.dim_model_base
                      << " hidden_size=" << meta.d
                      << " output_scale=" << output_scale << std::endl;
        }
        // #endregion agent log

        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, rsrc.w_out_norm, meta.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, rsrc.w_out_embd, output_scale, 0.0, nullptr, nullptr);

            auto log_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            logSoftmax(log_logits_buf, last_logits_buf);

            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, log_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        rsrc.w_out_norm,
                        meta.epsilon);
            }

            // #region agent log - Investigate hidden state and weight magnitudes
            static bool debug_investigate = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
            if (debug_investigate && nreq > 0 && (req_pos[0] == 0 || (req_pos[0] >= 1345 && req_pos[0] < 1380))) {
                RUN_INFINI(infinirtStreamSynchronize(stream));
                size_t last_token_idx = token_offset - 1;
                auto sample_size = std::min(100UL, d);
                auto hidden_before = std::vector<float>(sample_size);
                auto hidden_after = std::vector<float>(sample_size);
                auto weight_sample = std::vector<float>(sample_size);

                // Sample hidden states before RMSNorm
                if (dt_logits == INFINI_DTYPE_F32) {
                    RUN_INFINI(infinirtMemcpy(hidden_before.data(), logits_in->data(last_token_idx * d), sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
                    RUN_INFINI(infinirtMemcpy(hidden_after.data(), logits_out->slice(0, 0, 1)->data(), sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
                } else if (dt_logits == INFINI_DTYPE_F16) {
                    auto raw_before = std::vector<uint16_t>(sample_size);
                    auto raw_after = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw_before.data(), logits_in->data(last_token_idx * d), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    RUN_INFINI(infinirtMemcpy(raw_after.data(), logits_out->slice(0, 0, 1)->data(), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) {
                        hidden_before[i] = f16_to_f32(raw_before[i]);
                        hidden_after[i] = f16_to_f32(raw_after[i]);
                    }
                } else if (dt_logits == INFINI_DTYPE_BF16) {
                    auto raw_before = std::vector<uint16_t>(sample_size);
                    auto raw_after = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw_before.data(), logits_in->data(last_token_idx * d), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    RUN_INFINI(infinirtMemcpy(raw_after.data(), logits_out->slice(0, 0, 1)->data(), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) {
                        hidden_before[i] = bf16_to_f32(raw_before[i]);
                        hidden_after[i] = bf16_to_f32(raw_after[i]);
                    }
                }

                // Sample output embedding weights
                auto weight_dtype = rsrc.w_out_embd->dtype();
                if (weight_dtype == INFINI_DTYPE_F32) {
                    RUN_INFINI(infinirtMemcpy(weight_sample.data(), rsrc.w_out_embd->data(0), sample_size * sizeof(float), INFINIRT_MEMCPY_D2H));
                } else if (weight_dtype == INFINI_DTYPE_F16) {
                    auto raw = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw.data(), rsrc.w_out_embd->data(0), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) weight_sample[i] = f16_to_f32(raw[i]);
                } else if (weight_dtype == INFINI_DTYPE_BF16) {
                    auto raw = std::vector<uint16_t>(sample_size);
                    RUN_INFINI(infinirtMemcpy(raw.data(), rsrc.w_out_embd->data(0), sample_size * sizeof(uint16_t), INFINIRT_MEMCPY_D2H));
                    for (size_t i = 0; i < sample_size; i++) weight_sample[i] = bf16_to_f32(raw[i]);
                }

                float max_hb = *std::max_element(hidden_before.begin(), hidden_before.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
                float max_ha = *std::max_element(hidden_after.begin(), hidden_after.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });
                float max_w = *std::max_element(weight_sample.begin(), weight_sample.end(), [](float a, float b) { return std::abs(a) < std::abs(b); });

                std::cerr << "[INVESTIGATE] req_pos=" << req_pos[0]
                          << " hidden_before_rmsnorm_max_abs=" << std::abs(max_hb)
                          << " hidden_after_rmsnorm_max_abs=" << std::abs(max_ha)
                          << " output_embd_weight_max_abs=" << std::abs(max_w)
                          << " output_scale=" << output_scale << std::endl;
            }
            // #endregion agent log

            // #region agent log - Keep output_scale to compensate for removing scale_output from norm weights
            // We removed scale_output from norm weights, so we need to apply scaling in linear projection
            // This matches PyTorch: logits = lm_head(hidden_states / (hidden_size / dim_model_base))
            // #endregion agent log
            linear(prob_buf, logits_out->slice(0, 0, nreq), rsrc.w_out_embd, output_scale, 0.0, nullptr, nullptr);

            // Apply repetition penalty if needed
            if (repetition_penalty != nullptr) {
                // Check if any penalty != 1.0
                bool need_penalty = false;
                for (uint32_t req = 0; req < nreq; req++) {
                    if (repetition_penalty[req] != 1.0f) {
                        need_penalty = true;
                        break;
                    }
                }

                if (need_penalty) {
                    // Create boolean mask [nreq, dvoc] on host
                    // Use uint8_t instead of bool because std::vector<bool> doesn't have data() method
                    std::vector<uint8_t> mask_host(nreq * dvoc, 0);

                    // Extract previous tokens and set mask
                    // tokens array contains only NEW tokens (for embedding lookup - keep separate!)
                    // full_tokens array contains full token history (for repetition penalty mask only)
                    if (full_tokens != nullptr && full_ntok > 0) {
                        // Use full token history for repetition penalty mask
                        size_t req_token_start = 0;
                        for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
                            // Calculate where this request's tokens start in the full tokens array
                            // For single request (nreq=1), req_token_start should always be 0
                            // For multiple requests, accumulate the lengths of previous requests
                            if (req_idx > 0) {
                                req_token_start += req_pos[req_idx - 1] + req_lens[req_idx - 1];
                            }

                            // Mark all tokens from 0 to req_pos[req_idx] + req_lens[req_idx] - 1 (excluding last)
                            // total_seq_len = current position + new tokens in this batch
                            uint32_t total_seq_len = req_pos[req_idx] + req_lens[req_idx];
                            // actual_len = number of tokens to mark (all previous tokens, excluding the one being predicted)
                            uint32_t actual_len = total_seq_len - 1;

                            // Ensure we don't exceed the full_tokens array bounds
                            if (req_token_start + actual_len > full_ntok) {
                                actual_len = (req_token_start < full_ntok) ? (full_ntok - req_token_start) : 0;
                            }

                            // #region agent log - Trace mask creation (dynamic token tracking)
                            static bool debug_mask = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
                            bool should_trace = debug_mask && req_idx == 0 && (req_pos[req_idx] >= 1345 && req_pos[req_idx] < 1380);
                            // #endregion agent log

                            // Mark tokens in mask
                            for (uint32_t i = 0; i < actual_len; i++) {
                                if (req_token_start + i < full_ntok) {
                                    uint32_t token_id = full_tokens[req_token_start + i];
                                    if (token_id < dvoc) {
                                        mask_host[req_idx * dvoc + token_id] = 1;
                                    }
                                }
                            }

                            // #region agent log - Trace mask creation details
                            if (should_trace) {
                                std::cerr << "[MASK_TRACE] req_pos=" << req_pos[req_idx]
                                          << " req_lens=" << req_lens[req_idx]
                                          << " req_token_start=" << req_token_start
                                          << " total_seq_len=" << total_seq_len
                                          << " actual_len=" << actual_len
                                          << " full_ntok=" << full_ntok << std::endl;
                                // Count tokens in full_tokens range
                                std::map<uint32_t, uint32_t> token_counts;
                                for (uint32_t i = 0; i < actual_len && req_token_start + i < full_ntok; i++) {
                                    uint32_t token_id = full_tokens[req_token_start + i];
                                    if (token_id < dvoc) {
                                        token_counts[token_id]++;
                                    }
                                }
                                // Find most frequent token
                                uint32_t most_freq_token = 0;
                                uint32_t most_freq_count = 0;
                                for (const auto& pair : token_counts) {
                                    if (pair.second > most_freq_count) {
                                        most_freq_count = pair.second;
                                        most_freq_token = pair.first;
                                    }
                                }
                                std::cerr << "[MASK_TRACE] Most frequent token in range: " << most_freq_token
                                          << " (appears " << most_freq_count << " times)" << std::endl;
                                std::cerr << "[MASK_TRACE] mask_host[" << most_freq_token << "] = "
                                          << (int)mask_host[req_idx * dvoc + most_freq_token] << std::endl;
                                // Check last few tokens in full_tokens
                                uint32_t check_start = (req_token_start + actual_len > 10) ? (req_token_start + actual_len - 10) : req_token_start;
                                std::cerr << "[MASK_TRACE] Last 10 tokens in range: ";
                                for (uint32_t i = check_start; i < req_token_start + actual_len && i < full_ntok; i++) {
                                    std::cerr << full_tokens[i] << " ";
                                }
                                std::cerr << std::endl;
                            }
                            // #endregion agent log
                        }
                    } else {
                        // Fallback: only penalize tokens in current batch if full_tokens not available
                        size_t token_offset = 0;
                        for (uint32_t req_idx = 0; req_idx < nreq; req_idx++) {
                            if (req_lens[req_idx] > 1) {
                                for (uint32_t i = 0; i < req_lens[req_idx] - 1; i++) {
                                    uint32_t token_id = tokens[token_offset + i];
                                    if (token_id < dvoc) {
                                        mask_host[req_idx * dvoc + token_id] = 1;
                                    }
                                }
                            }
                            token_offset += req_lens[req_idx];
                        }
                    }

                    // Create mask tensor and copy from host
                    auto mask_buf = Tensor::buffer(INFINI_DTYPE_BOOL, {nreq, dvoc}, rsrc.memory_pool);
                    RUN_INFINI(infinirtMemcpyAsync(mask_buf->data(), mask_host.data(),
                                                  nreq * dvoc * sizeof(uint8_t),
                                                  INFINIRT_MEMCPY_H2D, stream));

                    // Allocate device buffer for penalties
                    auto d_penalties = Tensor::buffer(INFINI_DTYPE_F32, {nreq}, rsrc.memory_pool);
                    RUN_INFINI(infinirtMemcpyAsync(d_penalties->data(), repetition_penalty,
                                                  nreq * sizeof(float),
                                                  INFINIRT_MEMCPY_H2D, stream));

                    // Debug logging (enable with REPETITION_PENALTY_DEBUG=1)
                    static bool debug_enabled = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
                    bool should_log = false;
                    if (debug_enabled && nreq > 0) {
                        // Log at regular intervals to track logit magnitude over time
                        should_log = (req_pos[0] % 50 == 0) || (req_pos[0] >= 1345 && req_pos[0] < 1380);
                    }

                    // Sample logits before penalty to track magnitude
                    std::vector<float> logits_before;
                    std::vector<std::pair<float, uint32_t>> logit_pairs_before;  // Save for after-penalty comparison
                    uint32_t repeating_tokens[] = {18867, 6686, 6783, 59466, 59566};
                    if (should_log) {
                        RUN_INFINI(infinirtStreamSynchronize(stream));

                        // Read logits and convert to float based on dtype
                        auto prob_host = std::vector<float>(nreq * dvoc);
                        if (dt_logits == INFINI_DTYPE_F32) {
                            RUN_INFINI(infinirtMemcpy(prob_host.data(), prob_buf->data(),
                                                      nreq * dvoc * sizeof(float),
                                                      INFINIRT_MEMCPY_D2H));
                        } else {
                            auto raw_buf = std::vector<uint8_t>(nreq * dvoc * dsize(dt_logits));
                            RUN_INFINI(infinirtMemcpy(raw_buf.data(), prob_buf->data(),
                                                      nreq * dvoc * dsize(dt_logits),
                                                      INFINIRT_MEMCPY_D2H));
                            if (dt_logits == INFINI_DTYPE_F16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t h = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = f16_to_f32(h);
                                }
                            } else if (dt_logits == INFINI_DTYPE_BF16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t b = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = bf16_to_f32(b);
                                }
                            }
                        }

                        // Calculate logit statistics
                        float max_logit = prob_host[0];
                        float min_logit = prob_host[0];
                        float sum_logit = 0.0f;
                        for (size_t i = 0; i < nreq * dvoc; i++) {
                            if (prob_host[i] > max_logit) max_logit = prob_host[i];
                            if (prob_host[i] < min_logit) min_logit = prob_host[i];
                            sum_logit += prob_host[i];
                        }
                        float avg_logit = sum_logit / (nreq * dvoc);

                        // Find top 5 logits
                        logit_pairs_before.clear();
                        for (uint32_t i = 0; i < dvoc; i++) {
                            logit_pairs_before.push_back({prob_host[0 * dvoc + i], i});
                        }
                        std::sort(logit_pairs_before.begin(), logit_pairs_before.end(),
                                  [](const auto& a, const auto& b) { return a.first > b.first; });

                        for (uint32_t tidx = 0; tidx < 5; tidx++) {
                            uint32_t token_id = repeating_tokens[tidx];
                            if (token_id < dvoc) {
                                logits_before.push_back(prob_host[0 * dvoc + token_id]);
                            }
                        }

                        uint32_t mask_ones = 0;
                        for (uint32_t i = 0; i < nreq * dvoc; i++) {
                            if (mask_host[i] != 0) mask_ones++;
                        }

                        std::cerr << "[REP_PENALTY] req_pos=" << req_pos[0]
                                  << " logits: max=" << max_logit << " min=" << min_logit
                                  << " avg=" << avg_logit
                                  << " mask=" << mask_ones << " marked"
                                  << " penalty=" << repetition_penalty[0] << std::endl;

                        // Log top 5 logits to see which tokens are dominant
                        if (req_pos[0] >= 1345 && req_pos[0] < 1380) {
                            std::cerr << "  Top 5 logits BEFORE penalty: ";
                            for (uint32_t i = 0; i < 5 && i < logit_pairs_before.size(); i++) {
                                std::cerr << "token[" << logit_pairs_before[i].second << "]=" << logit_pairs_before[i].first << " ";
                            }
                            std::cerr << std::endl;
                        }
                    }

                    // #region agent log - Verify mask before applying penalty (dynamic top token tracking)
                    static bool debug_mask_before = (getenv("REPETITION_PENALTY_DEBUG") != nullptr);
                    if (debug_mask_before && nreq > 0 && (req_pos[0] >= 1345 && req_pos[0] < 1380)) {
                        RUN_INFINI(infinirtStreamSynchronize(stream));
                        // Read logits to find top token
                        auto prob_host = std::vector<float>(nreq * dvoc);
                        if (dt_logits == INFINI_DTYPE_F32) {
                            RUN_INFINI(infinirtMemcpy(prob_host.data(), prob_buf->data(),
                                                      nreq * dvoc * sizeof(float),
                                                      INFINIRT_MEMCPY_D2H));
                        } else {
                            auto raw_buf = std::vector<uint8_t>(nreq * dvoc * dsize(dt_logits));
                            RUN_INFINI(infinirtMemcpy(raw_buf.data(), prob_buf->data(),
                                                      nreq * dvoc * dsize(dt_logits),
                                                      INFINIRT_MEMCPY_D2H));
                            if (dt_logits == INFINI_DTYPE_F16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t h = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = f16_to_f32(h);
                                }
                            } else if (dt_logits == INFINI_DTYPE_BF16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t b = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = bf16_to_f32(b);
                                }
                            }
                        }
                        // Find top token
                        uint32_t top_token = 0;
                        float max_logit = prob_host[0];
                        for (uint32_t i = 1; i < dvoc; i++) {
                            if (prob_host[i] > max_logit) {
                                max_logit = prob_host[i];
                                top_token = i;
                            }
                        }
                        // Copy mask back from device to verify
                        std::vector<uint8_t> mask_check(nreq * dvoc);
                        RUN_INFINI(infinirtMemcpy(mask_check.data(), mask_buf->data(),
                                                  nreq * dvoc * sizeof(uint8_t),
                                                  INFINIRT_MEMCPY_D2H));
                        std::cerr << "[MASK_VERIFY] req_pos=" << req_pos[0]
                                  << " top_token=" << top_token
                                  << " top_logit=" << max_logit
                                  << " mask[" << top_token << "]=" << (int)mask_check[top_token] << std::endl;
                    }
                    // #endregion agent log

                    // Apply repetition penalty
                    applyRepetitionPenalty(prob_buf, mask_buf,
                                          static_cast<const float *>(d_penalties->data()));

                    // Sample logits after penalty
                    if (should_log) {
                        RUN_INFINI(infinirtStreamSynchronize(stream));

                        auto prob_host = std::vector<float>(nreq * dvoc);
                        if (dt_logits == INFINI_DTYPE_F32) {
                            RUN_INFINI(infinirtMemcpy(prob_host.data(), prob_buf->data(),
                                                      nreq * dvoc * sizeof(float),
                                                      INFINIRT_MEMCPY_D2H));
                        } else {
                            auto raw_buf = std::vector<uint8_t>(nreq * dvoc * dsize(dt_logits));
                            RUN_INFINI(infinirtMemcpy(raw_buf.data(), prob_buf->data(),
                                                      nreq * dvoc * dsize(dt_logits),
                                                      INFINIRT_MEMCPY_D2H));
                            if (dt_logits == INFINI_DTYPE_F16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t h = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = f16_to_f32(h);
                                }
                            } else if (dt_logits == INFINI_DTYPE_BF16) {
                                for (size_t i = 0; i < nreq * dvoc; i++) {
                                    uint16_t b = *reinterpret_cast<uint16_t*>(&raw_buf[i * 2]);
                                    prob_host[i] = bf16_to_f32(b);
                                }
                            }
                        }

                        // #region agent log - Log top 5 tokens before/after penalty
                        if (req_pos[0] >= 1345 && req_pos[0] < 1380) {
                            // Find top 5 tokens after penalty
                            std::vector<std::pair<float, uint32_t>> logit_pairs_after;
                            for (uint32_t i = 0; i < dvoc; i++) {
                                logit_pairs_after.push_back({prob_host[0 * dvoc + i], i});
                            }
                            std::sort(logit_pairs_after.begin(), logit_pairs_after.end(),
                                      [](const auto& a, const auto& b) { return a.first > b.first; });

                            std::cerr << "[REP_PENALTY_AFTER] req_pos=" << req_pos[0] << " Top 5 tokens AFTER penalty:" << std::endl;
                            for (uint32_t i = 0; i < 5 && i < logit_pairs_after.size(); i++) {
                                uint32_t token_id = logit_pairs_after[i].second;
                                float logit_after = logit_pairs_after[i].first;
                                // Find corresponding before value from logit_pairs
                                float logit_before = 0.0f;
                                for (const auto& pair : logit_pairs_before) {
                                    if (pair.second == token_id) {
                                        logit_before = pair.first;
                                        break;
                                    }
                                }
                                bool is_marked = (mask_host[0 * dvoc + token_id] != 0);
                                float change = logit_after - logit_before;
                                std::cerr << "  token[" << token_id << "]: " << logit_before
                                          << " -> " << logit_after << " (Δ" << change << ") "
                                          << (is_marked ? "[MARKED]" : "[NOT MARKED]") << std::endl;
                            }
                        }
                        // #endregion agent log

                        std::cerr << "[REP_PENALTY] Operator called. Logit changes:" << std::endl;
                        for (uint32_t tidx = 0; tidx < 5; tidx++) {
                            uint32_t token_id = repeating_tokens[tidx];
                            if (token_id < dvoc && tidx < logits_before.size()) {
                                float logit_after = prob_host[0 * dvoc + token_id];
                                float logit_before = logits_before[tidx];
                                float change = logit_after - logit_before;
                                bool is_marked = (mask_host[0 * dvoc + token_id] != 0);

                                std::cerr << "  Token " << token_id
                                          << ": " << logit_before << " -> " << logit_after
                                          << " (Δ" << change << ")"
                                          << (is_marked ? " [MARKED]" : " [NOT MARKED]") << std::endl;
                            }
                        }
                    }
                }
            }

            // #region agent log - Use fixed seed for deterministic testing
            static bool use_fixed_seed = (getenv("FIXED_RANDOM_SEED") != nullptr);
            static thread_local std::mt19937 gen(use_fixed_seed ? 42 : std::random_device{}());
            // #endregion agent log
            token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
                randomSample(result_buf->slice(0, req, 1)->view_as({}, {}),
                             prob_buf->slice(0, req, 1)->view_as({dvoc}, {1}),
                             random_val, topp[req], topk[req], temperature[req]);
                token_offset += seq_len;
            }
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = uint32_t(result_cpu[req]);
            }
        }
    }
}

__C void
inferBatchJiuge(struct JiugeModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                const float *repetition_penalty,
                const uint32_t *full_tokens, uint32_t full_ntok,
                uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;
    model->req.repetition_penalty = repetition_penalty;
    model->req.full_tokens = full_tokens;
    model->req.full_ntok = full_ntok;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatchJiuge(struct JiugeModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const int32_t *block_tables,
           const int32_t *slot_mapping,
           const float *temperature, const uint32_t *topk, const float *topp,
           const uint32_t is_prefill, const bool enable_paged_attn,
           uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.block_tables = block_tables;
    model->req.slot_mapping = slot_mapping;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;
    model->req.is_prefill = is_prefill;
    model->req.enable_paged_attn = enable_paged_attn;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
forwardBatch(struct JiugeModel *model,
             const uint32_t *tokens, uint32_t ntok,
             const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
             struct KVCache **kv_caches,
             const int32_t *block_tables,
             const int32_t *slot_mapping,
             const uint32_t is_prefill, const bool enable_paged_attn,
             void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.block_tables = block_tables;
    model->req.slot_mapping = slot_mapping;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;
    model->req.is_prefill = is_prefill;
    model->req.enable_paged_attn = enable_paged_attn;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, JiugeDeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {

    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;
        }

        bool enable_paged = meta.kvcache_block_size != 0;
        if (enable_paged){
            inferDeviceBatchPaged(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                req.block_tables, req.slot_mapping,
                req.temperature, req.topk, req.topp,
                req.repetition_penalty,
                req.full_tokens, req.full_ntok,
                req.is_prefill, req.enable_paged_attn,
                req.output, req.logits);
        }
        else{
            inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                req.temperature, req.topk, req.topp,
                req.repetition_penalty,
                req.full_tokens, req.full_ntok,
                req.output, req.logits);

        }

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<JiugeDeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {

        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct JiugeModel *
createJiugeModel(const JiugeMeta *meta,
                 const JiugeWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    JiugeModel *model = new JiugeModel(meta, weights, device, device_ids);
    return model;
}

__C void destroyJiugeModel(struct JiugeModel *model) {
    auto ndev = model->dev_resources.size();

    for (size_t idev = 0; idev < ndev; idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].exit_flag = true;
        lock.unlock();
        model->states[idev].cv_start.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}
