#include "qwen_hybrid.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"

#include <random>
#include <thread>
#include <vector>

inline void createDeviceResource(DeviceResource *rsrc, const QwenHybridMeta *meta,
                                 std::shared_ptr<QwenHybridDeviceWeight> weights,
                                 infiniDevice_t device, int idev,
                                 int ndev, int dev_id,
                                 infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
        device,
        dev_id,
        handle,
        weights,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

inline void releaseDeviceResource(DeviceResource &res) {
    infinirtDeviceSynchronize();
    // Release individual Tensors

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

inline void fix_query_key_value_ordering() {
}

void inferDeviceBatch(const QwenHybridMeta *meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      struct MambaCache **mamba_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits) {
    auto nlayer = meta->nlayer;
    auto nkvh = meta->nkvh / ndev;
    auto nh = meta->nh / ndev;
    auto ngroup = nh / nkvh;
    bool use_qk_norm = meta->use_qk_norm;
    // auto dctx = meta.dctx;
    auto dh = meta->dh;
    auto d = meta->d;
    auto dt_logits = meta->dtype;
    //  auto di = meta->shared_di / ndev;
    auto dvoc = meta->dvoc;
    auto stream = rsrc.stream;
    auto weight = rsrc.weights;

    // linear
    // bool use_precomputed_state = false;
    auto conv_kernel_dim = meta->l_conv_kernel_dim;
    auto head_k_dim = meta->l_k_dim;
    auto head_v_dim = meta->l_v_dim;

    auto num_v_heads = meta->l_n_v_head / ndev;
    auto num_k_heads = meta->l_n_k_head / ndev;

    auto key_dim = head_k_dim * num_k_heads;
    auto value_dim = head_v_dim * num_v_heads;

    // Allocate buffers
    auto logits_in
        = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto k_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);
    auto v_buf = Tensor::buffer(dt_logits, {ntok, nkvh * dh}, rsrc.memory_pool);

    // auto gate_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
    // auto up_buf = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);

    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // linear buffer
    auto projected_states_qkvz_buf = Tensor::buffer(dt_logits, {ntok, key_dim * 2 + value_dim * 2}, rsrc.memory_pool);
    auto projected_states_ba_buf = Tensor::buffer(dt_logits, {ntok, num_v_heads * 2}, rsrc.memory_pool);
    // auto b_buf = projected_states_ba_buf->view({ntok, num_k_heads, num_v_heads / num_k_heads * 2})->slice(2, 0, num_v_heads / num_k_heads);
    // auto a_buf = projected_states_ba_buf->view({ntok, num_k_heads, num_v_heads / num_k_heads * 2})->slice(2, num_v_heads / num_k_heads, num_v_heads / num_k_heads);
    auto b_buf = Tensor::buffer(dt_logits, {ntok, num_k_heads, num_v_heads / num_k_heads}, rsrc.memory_pool);
    auto a_buf = Tensor::buffer(dt_logits, {ntok, num_k_heads, num_v_heads / num_k_heads}, rsrc.memory_pool);

    auto z = Tensor::buffer(dt_logits, {ntok, value_dim}, rsrc.memory_pool);
    auto linear_attn_o_buf = Tensor::buffer(dt_logits, {ntok, value_dim}, rsrc.memory_pool);

    auto alpha_la_g = Tensor::buffer(dt_logits, weight->alpha_la_g[0]->shape(), rsrc.memory_pool);

    // Prepare inputs
    auto batch_pos_ids
        = std::vector<uint32_t>(ntok);
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
                                       weight->w_in_embd->data(tokens[i] * d),
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
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh * ngroup * max_seq_len * dh}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // Compute
    size_t attn_cache_layer = 0,
           linear_cache_layer = 0;
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weight->w_attn_norm[layer], meta->epsilon);

        if (layer % 4 == 3) {
            // qkv_proj && qkv_bias
            std::shared_ptr<Tensor> b_attn_q;
            std::shared_ptr<Tensor> b_attn_k;
            std::shared_ptr<Tensor> b_attn_v;
            if (weight->b_attn_q.size() > 0) {
                b_attn_q = weight->b_attn_q[layer];
                b_attn_k = weight->b_attn_k[layer];
                b_attn_v = weight->b_attn_v[layer];
            }

            linear(q_buf, logits_out, weight->w_attn_q[layer], 1.0, 0.0, nullptr, b_attn_q ? b_attn_q : nullptr);
            linear(k_buf, logits_out, weight->w_attn_k[layer], 1.0, 0.0, nullptr, b_attn_k ? b_attn_k : nullptr);
            linear(v_buf, logits_out, weight->w_attn_v[layer], 1.0, 0.0, nullptr, b_attn_v ? b_attn_v : nullptr);
            if (use_qk_norm) {
                rmsnorm(q_buf->view({ntok, nh, dh}), q_buf->view({ntok, nh, dh}), weight->w_attn_q_norm[layer], meta->epsilon);
                rmsnorm(k_buf->view({ntok, nkvh, dh}), k_buf->view({ntok, nkvh, dh}), weight->w_attn_k_norm[layer], meta->epsilon);
            }
            // rope
            rope_v2(q_buf->view({ntok, nh, dh}), q_buf->view({ntok, nh, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
            rope_v2(k_buf->view({ntok, nkvh, dh}), k_buf->view({ntok, nkvh, dh}), pos_ids_buf, weight->sin_table, weight->cos_table);
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto past_len = req_pos[req];
                auto seq_len = req_lens[req];
                auto total_len = past_len + seq_len;
                auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
                auto k = k_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});
                auto v = v_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, dh});

                // self attention
                // concat
                rearrange(kv_caches[req]->k[idev][attn_cache_layer]->slice(0, past_len, seq_len), k);
                rearrange(kv_caches[req]->v[idev][attn_cache_layer]->slice(0, past_len, seq_len), v);
                // qk
                auto q_rearrange = rearrange_q_buf->slice(0, 0, nkvh * ngroup * seq_len * dh)->view({nkvh, ngroup, seq_len, dh});
                rearrange(q_rearrange, q);
                auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
                auto k_gemm = kv_caches[req]->k[idev][attn_cache_layer]->slice(0, 0, total_len)->permute({1, 2, 0});
                linear(qk_gemm, q_rearrange->view({nkvh, ngroup * seq_len, dh}), k_gemm, 1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);
                // softmax
                auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
                causalSoftmax(qk_softmax, qk_softmax);
                auto v_gemm = kv_caches[req]->v[idev][attn_cache_layer]->slice(0, 0, total_len)->permute({1, 0, 2});
                linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
                // rearrange attn val
                rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

                token_offset += seq_len;
            }
            // o_proj
            linear(logits_in, o_buf, weight->w_attn_out[layer],
                   1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
            attn_cache_layer += 1;
        } else {
            linear(projected_states_qkvz_buf, logits_out, weight->w_la_qkvz[layer], 1.0, 0.0, nullptr, nullptr);
            linear(projected_states_ba_buf, logits_out, weight->w_la_ba[layer], 1.0, 0.0, nullptr, nullptr);

            rearrange(b_buf, projected_states_ba_buf->view({ntok, num_k_heads, num_v_heads / num_k_heads * 2})->slice(2, 0, num_v_heads / num_k_heads));
            rearrange(a_buf, projected_states_ba_buf->view({ntok, num_k_heads, num_v_heads / num_k_heads * 2})->slice(2, num_v_heads / num_k_heads, num_v_heads / num_k_heads));
            sigmoid(b_buf, b_buf);
            add(a_buf, a_buf, weight->b_la_dt[layer]->view({num_k_heads, num_v_heads / num_k_heads})->insertBroadcastDim(0, ntok));
            softplus(a_buf, a_buf);
            mul(a_buf, a_buf, weight->alpha_la_g[layer]->view({num_k_heads, num_v_heads / num_k_heads})->insertBroadcastDim(0, ntok));

            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                auto conv_state = mamba_caches[req]->conv_states[idev][linear_cache_layer];
                auto recurrent_state = mamba_caches[req]->ssm_states[idev][linear_cache_layer];

                // slice out qkv of current req and reshape to conv1d input
                auto mixed_qkv = Tensor::buffer(dt_logits, {key_dim * 2 + value_dim, conv_kernel_dim - 1 + seq_len}, rsrc.memory_pool);
                auto qkvz_reorder = projected_states_qkvz_buf
                                        ->view({ntok, num_k_heads, (key_dim * 2 + value_dim * 2) / num_k_heads})
                                        ->slice({{0, token_offset, seq_len}, {2, 0, (key_dim * 2 + value_dim) / num_k_heads}})
                                        ->permute({1, 2, 0});
                auto mixed_qkv_reorder = mixed_qkv->slice(1, conv_kernel_dim - 1, seq_len);
                rearrange(mixed_qkv_reorder
                              ->slice(0, 0, key_dim)
                              ->view({num_k_heads, key_dim / num_k_heads, seq_len}),
                          qkvz_reorder->slice({{1, 0, head_k_dim}}));
                rearrange(mixed_qkv_reorder
                              ->slice(0, key_dim, key_dim)
                              ->view({num_k_heads, key_dim / num_k_heads, seq_len}),
                          qkvz_reorder->slice(1, head_k_dim, head_k_dim));
                rearrange(mixed_qkv_reorder
                              ->slice(0, 2 * key_dim, value_dim)
                              ->view({num_k_heads, value_dim / num_k_heads, seq_len}),
                          qkvz_reorder->slice(1, head_k_dim * 2, value_dim / num_k_heads));

                // concat previous conv state as padding
                rearrange(mixed_qkv->slice(1, 0, conv_kernel_dim - 1),
                          conv_state);
                rearrange(conv_state, mixed_qkv->slice(1, seq_len, conv_kernel_dim - 1));
                mixed_qkv = mixed_qkv->view({1, key_dim * 2 + value_dim, conv_kernel_dim - 1 + seq_len});
                auto mixed_qkv_out = Tensor::buffer(dt_logits, {1, key_dim * 2 + value_dim, seq_len}, rsrc.memory_pool);
                //---------------- conv1d
                conv1d(mixed_qkv_out, mixed_qkv, weight->w_la_conv[layer], nullptr, nullptr, nullptr, nullptr, key_dim * 2 + value_dim);
                silu(mixed_qkv_out, mixed_qkv_out);

                // q k v z
                auto query = Tensor::buffer(dt_logits, {1, num_v_heads, seq_len, head_k_dim}, rsrc.memory_pool);
                auto key = Tensor::buffer(dt_logits, {1, num_v_heads, seq_len, head_k_dim}, rsrc.memory_pool);
                auto value = Tensor::buffer(dt_logits, {1, num_v_heads, seq_len, head_v_dim}, rsrc.memory_pool);
                // transpose to {seq_len, key_dim * 2 + value_dim}
                mixed_qkv_out = mixed_qkv_out->view({key_dim * 2 + value_dim, seq_len});
                // broadcast num_qk_heads to num_v_heads
                rearrange(query->view({num_k_heads, num_v_heads / num_k_heads, seq_len, head_k_dim}),
                          mixed_qkv_out
                              ->slice(0, 0, key_dim)
                              ->view({num_k_heads, head_k_dim, seq_len})
                              ->permute({0, 2, 1})
                              ->insertBroadcastDim(1, num_v_heads / num_k_heads));
                rearrange(key->view({num_k_heads, num_v_heads / num_k_heads, seq_len, head_k_dim}),
                          mixed_qkv_out
                              ->slice(0, key_dim, key_dim)
                              ->view({num_k_heads, head_k_dim, seq_len})
                              ->permute({0, 2, 1})
                              ->insertBroadcastDim(1, num_v_heads / num_k_heads));
                rearrange(value->view({num_v_heads, seq_len, head_v_dim}),
                          mixed_qkv_out
                              ->slice(0, key_dim * 2, value_dim)
                              ->view({num_v_heads, head_v_dim, seq_len})
                              ->permute({0, 2, 1}));
                // get beta and g
                auto beta = Tensor::buffer(dt_logits, {num_v_heads, seq_len}, rsrc.memory_pool);
                auto g = Tensor::buffer(dt_logits, {num_v_heads, seq_len}, rsrc.memory_pool);
                rearrange(beta->view({num_k_heads, num_v_heads / num_k_heads, seq_len}), b_buf->slice(0, token_offset, seq_len)->permute({1, 2, 0}));
                rearrange(g->view({num_k_heads, num_v_heads / num_k_heads, seq_len}), a_buf->slice(0, token_offset, seq_len)->permute({1, 2, 0}));

                auto linear_attn_out = Tensor::buffer(dt_logits, {num_v_heads, seq_len, head_v_dim}, rsrc.memory_pool);

                if (seq_len > 1) {
                    size_t chunk_size = 8;
                    chunk_gated_delta_rule(
                        linear_attn_out->view({1, num_v_heads, seq_len, head_v_dim}),
                        recurrent_state->view({1, num_v_heads, head_k_dim, head_v_dim}),
                        query,
                        key,
                        value,
                        g->view({1, num_v_heads, seq_len}),
                        beta->view({1, num_v_heads, seq_len}),
                        recurrent_state->view({1, num_v_heads, head_k_dim, head_v_dim}),
                        true,
                        chunk_size);
                } else {
                    recurrent_gated_delta_rule(
                        linear_attn_out->view({1, num_v_heads, seq_len, head_v_dim}),
                        recurrent_state->view({1, num_v_heads, head_k_dim, head_v_dim}),
                        query,
                        key,
                        value,
                        g->view({1, num_v_heads, seq_len}),
                        beta->view({1, num_v_heads, seq_len}),
                        recurrent_state->view({1, num_v_heads, head_k_dim, head_v_dim}),
                        true);
                }

                rearrange(linear_attn_o_buf->slice(0, token_offset, seq_len)->view({seq_len, num_v_heads, head_v_dim}),
                          linear_attn_out->permute({1, 0, 2}));

                token_offset += seq_len;
            }
            auto z_buf = projected_states_qkvz_buf
                             ->view({ntok, num_k_heads, (key_dim * 2 + value_dim * 2) / num_k_heads})
                             ->slice(2, (key_dim * 2 + value_dim) / num_k_heads, value_dim / num_k_heads);
            rearrange(z->view({ntok, num_k_heads, value_dim / num_k_heads}), z_buf);
            gated_rmsnorm(linear_attn_o_buf->view({ntok, num_v_heads, head_v_dim}),
                          linear_attn_o_buf->view({ntok, num_v_heads, head_v_dim}),
                          weight->w_la_norm[layer],
                          z->view({ntok, num_v_heads, head_v_dim}),
                          meta->epsilon);
            linear(logits_in, linear_attn_o_buf, weight->w_la_out[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
            linear_cache_layer += 1;
        }

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 2. FFN
        rmsnorm(logits_out, logits_in, weight->w_ffn_norm[layer], meta->epsilon);

        // MLP
        {
            // linear(gate_buf, logits_out, weight->w_ffn_gate[layer], 1.0, 0.0, nullptr, nullptr);
            // linear(up_buf, logits_out, weight->w_ffn_up[layer], 1.0, 0.0, nullptr, nullptr);
            // swiglu(gate_buf, up_buf, gate_buf);
            // linear(logits_in, gate_buf, weight->w_ffn_down[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        }

        // ------------------------------------------------------------------------ //
        //                          SparseMLP                                       //
        // ------------------------------------------------------------------------ //
        {
            // 输入变量:
            std::shared_ptr<Tensor> hidden_states = logits_out; // logits_out 是整个 MoE的输入，重新起名字为 hidden_states

            // 需要提前申请的缓存
            size_t moe_intermediate_size = meta->moe_di / ndev;

            auto router_gate_buf = Tensor::buffer(dt_logits, {1, 1 * moe_intermediate_size}, rsrc.memory_pool);
            auto router_up_buf = Tensor::buffer(dt_logits, {1, 1 * moe_intermediate_size}, rsrc.memory_pool);

            size_t shared_expert_intermediate_size = meta->shared_di / ndev;
            auto shared_gate_buf = Tensor::buffer(dt_logits, {ntok, 1 * shared_expert_intermediate_size}, rsrc.memory_pool);
            auto shared_up_buf = Tensor::buffer(dt_logits, {ntok, 1 * shared_expert_intermediate_size}, rsrc.memory_pool);

            std::shared_ptr<Tensor> shared_gate_output = Tensor::buffer(dt_logits, {ntok, 1}, rsrc.memory_pool);              // 共享专家的 gate 权重
            std::shared_ptr<Tensor> router_gate_output = Tensor::buffer(dt_logits, {ntok, meta->nexperts}, rsrc.memory_pool); // 路由专家的 gate 的输出

            // 需要提前申请的缓存
            std::shared_ptr<Tensor> router_states_sum = Tensor::buffer(hidden_states->dtype(), hidden_states->shape(), rsrc.memory_pool); // 用于存储 router MLP的输出
            std::shared_ptr<Tensor> shared_states = Tensor::buffer(hidden_states->dtype(), hidden_states->shape(), rsrc.memory_pool);     // 用于存储 shared MLP的输出

            size_t topk = meta->kexperts;
            bool norm_topk_prob = meta->norm_topk_prob;

            std::shared_ptr<Tensor> values_gpu = Tensor::buffer(infiniDtype_t::INFINI_DTYPE_F32, {ntok, topk}, rsrc.memory_pool);  // 用于存储topkrouter的输出，每个expert对应的加权权重。
            std::shared_ptr<Tensor> indices_gpu = Tensor::buffer(infiniDtype_t::INFINI_DTYPE_I32, {ntok, topk}, rsrc.memory_pool); // 用于存储topkrouter的输出，要经过哪些专家id（从256个中选8个）
            std::vector<float> values_cpu(ntok * topk, 0.f);
            std::vector<int> indices_cpu(ntok * topk, 0);

            // ------------------------------------------------------------------------ //
            //                            开始计算                                       //
            // ------------------------------------------------------------------------ //
            // (1) 共享专家：
            //      hidden_states 经过一个共享专家
            {
                // 输入: hidden_states
                // 输出: shared_states
                auto w_gate = weight->w_shared_expert_ffn_gate[layer];
                auto w_up = weight->w_shared_expert_ffn_up[layer];
                auto w_down = weight->w_shared_expert_ffn_down[layer];

                linear(shared_gate_buf, hidden_states, w_gate, 1.0, 0.0, nullptr, nullptr);
                linear(shared_up_buf, hidden_states, w_up, 1.0, 0.0, nullptr, nullptr);
                swiglu(shared_gate_buf, shared_up_buf, shared_gate_buf);
                linear(shared_states, shared_gate_buf, w_down, 1.0, 0.0, nullptr, nullptr); // only rank 0 adds residual

                // gate
                w_gate = weight->w_shared_expert_gate[layer];
                linear(shared_gate_output, hidden_states, w_gate, 1.0, 0.0, nullptr, nullptr); // N,1
                sigmoid(shared_gate_output, shared_gate_output);

                // shared的通信
                if (rsrc.comm != nullptr) {
                    RUN_INFINI(infinicclAllReduce(
                        shared_states->data(), shared_states->data(), ntok * d, dt_logits,
                        INFINICCL_SUM, rsrc.comm, stream));
                    RUN_INFINI(infinirtStreamSynchronize(stream));
                }

                // mul
                shared_gate_output = shared_gate_output->view_as(shared_states->shape(), {1, 0});
                mul(shared_states, shared_gate_output, shared_states);
            }

            // (2) topk操作：
            //      hidden_states 先经过 gate_weight，得到 router_gate_output
            //      router_gate_output 进行 topk 操作
            {
                auto w_gate = weight->w_router_expert_gate[layer];
                linear(router_gate_output, hidden_states, w_gate, 1.0, 0.0, nullptr, nullptr);

                topksoftmax(values_gpu, indices_gpu, router_gate_output, topk, norm_topk_prob);
                RUN_INFINI(infinirtMemcpy((void *)values_cpu.data(), values_gpu->data(), values_cpu.size() * sizeof(float), INFINIRT_MEMCPY_D2H));
                RUN_INFINI(infinirtMemcpy((void *)indices_cpu.data(), indices_gpu->data(), indices_cpu.size() * sizeof(int), INFINIRT_MEMCPY_D2H));
                RUN_INFINI(infinirtStreamSynchronize(rsrc.stream));
            }

            // (3) MoE操作：  每个 token 经过 4个 路由专家
            {
                // 输入: hidden_states, values_cpu，indices_cpu
                // 输出: 每个token的router专家加权和

                for (size_t itok = 0; itok < ntok; ++itok) {
                    std::shared_ptr<Tensor> hidden_states_i = hidden_states->slice(0, itok, 1);
                    std::shared_ptr<Tensor> router_states_sum_i = router_states_sum->slice(0, itok, 1);

                    // 经过第一个专家 : C = alpha * AB
                    {
                        int index = indices_cpu[itok * topk + 0];
                        float alpha = values_cpu[itok * topk + 0];

                        auto w_gate = weight->w_router_expert_ffn_gate[layer][index];
                        auto w_up = weight->w_router_expert_ffn_up[layer][index];
                        auto w_down = weight->w_router_expert_ffn_down[layer][index];

                        linear(router_gate_buf, hidden_states_i, w_gate, 1.0, 0.0, nullptr, nullptr);
                        linear(router_up_buf, hidden_states_i, w_up, 1.0, 0.0, nullptr, nullptr);
                        swiglu(router_gate_buf, router_up_buf, router_gate_buf);
                        linear(router_states_sum_i, router_gate_buf, w_down, alpha, 0.0, nullptr, nullptr); // only rank 0 adds residual
                    }

                    // 经过后续的专家 : C  = alpha * AB + C_last
                    for (size_t k = 1; k < topk; ++k) {

                        int index = indices_cpu[itok * topk + k];
                        float alpha = values_cpu[itok * topk + k];

                        auto w_gate = weight->w_router_expert_ffn_gate[layer][index];
                        auto w_up = weight->w_router_expert_ffn_up[layer][index];
                        auto w_down = weight->w_router_expert_ffn_down[layer][index];

                        linear(router_gate_buf, hidden_states_i, w_gate, 1.0, 0.0, nullptr, nullptr);
                        linear(router_up_buf, hidden_states_i, w_up, 1.0, 0.0, nullptr, nullptr);
                        swiglu(router_gate_buf, router_up_buf, router_gate_buf);
                        linear(router_states_sum_i, router_gate_buf, w_down, alpha, 0.0, router_states_sum_i, nullptr); // only rank 0 adds residual
                    }
                }

                if (rsrc.comm != nullptr) { // 通信2
                    RUN_INFINI(infinicclAllReduce(
                        router_states_sum->data(), router_states_sum->data(), ntok * d, dt_logits,
                        INFINICCL_SUM, rsrc.comm, stream));
                    RUN_INFINI(infinirtStreamSynchronize(stream));
                }
            }

            // (4) 两类专家相加
            add(router_states_sum, router_states_sum, shared_states); // 输出 ok

            // (5) 最后的残差连接
            add(logits_in, router_states_sum, logits_in);
        }

        // All_reduce if distributed
        // if (rsrc.comm != nullptr) {
        //     RUN_INFINI(infinicclAllReduce(
        //         logits_in->data(), logits_in->data(), ntok * d, dt_logits,
        //         INFINICCL_SUM, rsrc.comm, stream));
        //     RUN_INFINI(infinirtStreamSynchronize(stream));
        // }
    }
    // Sample and Output
    if (idev == 0) {
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, weight->w_out_norm, meta->epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(), dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }
        if (output != nullptr) {
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        weight->w_out_norm,
                        meta->epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), weight->w_out_embd, 1.0, 0.0, nullptr, nullptr);
            std::random_device _rd;
            std::mt19937 gen(_rd());
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
inferBatchQwenHybrid(struct QwenHybridModel *model,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct KVCache **kv_caches,
                     struct MambaCache **mamba_caches,
                     const float *temperature, const uint32_t *topk, const float *topp,
                     uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.mamba_caches = mamba_caches;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

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

void launchDevice(const QwenHybridMeta *meta, std::shared_ptr<QwenHybridDeviceWeight> weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    createDeviceResource(rsrc, meta, weights, device, idev, ndev, dev_id, comm);

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

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.mamba_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

QwenHybridModel::QwenHybridModel(const QwenHybridMeta *meta, const ModelWeights *weights_) {
    auto weights = (QwenHybridWeights *)(weights_);
    device = weights->device();
    dev_ids = weights->devIds();
    int ndev = int(dev_ids.size());
    dev_resources = std::vector<DeviceResource>(ndev);
    states = std::vector<InferState>(ndev);
    threads.resize(ndev);

    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    for (int i = 0; i < ndev; i++) {
        threads[i] = std::thread(launchDevice, meta, weights->device_weights()[i], &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
    }
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct QwenHybridModel *
createQwenHybridModel(const QwenHybridMeta *meta,
                      const ModelWeights *weights) {
    QwenHybridModel *model = new QwenHybridModel(meta, weights);
    return model;
}

__C void destroyQwenHybridModel(struct QwenHybridModel *model) {
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
