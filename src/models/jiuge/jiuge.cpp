#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <cstring>
#include <random>
#include <thread>
#include <vector>

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

static void inferDeviceBatchEx(const JiugeMeta &meta, JiugeDeviceResource &rsrc,
                               uint32_t idev, uint32_t ndev,
                               const uint32_t *tokens, uint32_t ntok,
                               const uint32_t *req_lens, uint32_t nreq,
                               const uint32_t *req_pos,
                               const uint32_t *kv_pos,
                               struct KVCache **kv_caches,
                               uint32_t n_override,
                               const uint32_t *override_pos,
                               const void *override_embeds,
                               const float *temperature, const uint32_t *topk, const float *topp,
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

    const char *override_ptr = reinterpret_cast<const char *>(override_embeds);
    const size_t unit = dsize(dt_logits);
    uint32_t override_idx = 0;
    for (uint32_t i = 0; i < ntok; i++) {
        if (override_ptr != nullptr && override_idx < n_override && override_pos[override_idx] == i) {
            void *dst = logits_in->data(i * d);
            const void *src = override_ptr + static_cast<size_t>(override_idx) * d * unit;
            if (rsrc.device == INFINI_DEVICE_CPU) {
                std::memcpy(dst, src, unit * d);
            } else {
                RUN_INFINI(infinirtMemcpyAsync(dst, src, unit * d, INFINIRT_MEMCPY_H2D, stream));
            }
            override_idx++;
            continue;
        }
        RUN_INFINI(infinirtMemcpyAsync(
            logits_in->data(i * d),
            rsrc.w_in_embd->data(tokens[i] * d),
            unit * d,
            INFINIRT_MEMCPY_D2D,
            stream));
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = kv_pos[req];
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
            auto past_len = kv_pos[req];
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
        if (last_logits != nullptr) {
            rmsnorm(logits_out, logits_in, rsrc.w_out_norm, meta.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
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
                        rsrc.w_out_norm,
                        meta.epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
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


// 同步机制：
// 条件变量：
// cv_start：主线程通知工作线程开始推理
// cv_done：工作线程通知主线程推理完成
// cv_load：工作线程通知主线程设备已加载完成
// 互斥锁：保护共享状态（proceed、loaded等标志位）

// 执行流程：
// 1. 创建模型 → 启动N个工作线程 → 线程等待cv_start信号
// 2. 调用inferBatchJiuge → 设置参数 → 发cv_start信号
// 3. 工作线程被唤醒 → 调用inferDeviceBatch → 发cv_done信号
// 4. 主线程等待所有cv_done → 推理完成

// inferBatchJiuge本身不执行具体的矩阵运算，它是一个协调器，负责：
// 准备推理参数
// 唤醒所有设备的工作线程
// 等待所有线程完成推理


__C void
inferBatchJiuge(struct JiugeModel *model,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output) {
    // 1. 设置推理参数（共享的请求结构体）
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = 0;
    model->req.override_pos = nullptr;
    model->req.override_embeds = nullptr;
    model->req.output = output;
    model->req.logits = nullptr;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    // 2. 通知所有设备线程开始工作
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;  // 设置信号
        lock.unlock();
        model->states[idev].cv_start.notify_one();  // 唤醒线程
    }
    
    // 3. 等待所有设备线程完成工作
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
inferBatchJiugeWithLogits(struct JiugeModel *model,
                         const uint32_t *tokens, uint32_t ntok,
                         const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                         struct KVCache **kv_caches,
                         const float *temperature, const uint32_t *topk, const float *topp,
                         uint32_t *output, void *logits) {
    // 1. 设置推理参数（共享的请求结构体）
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = 0;
    model->req.override_pos = nullptr;
    model->req.override_embeds = nullptr;
    model->req.output = output;
    model->req.logits = logits;  // 关键：设置 logits 输出
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    // 2. 通知所有设备线程开始工作
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;  // 设置信号
        lock.unlock();
        model->states[idev].cv_start.notify_one();  // 唤醒线程
    }

    // 3. 等待所有设备线程完成工作
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        lock.unlock();
    }
}

__C void
inferBatchJiugeEx(struct JiugeModel *model,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq,
                  const uint32_t *req_pos,
                  const uint32_t *kv_pos,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = kv_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = 0;
    model->req.override_pos = nullptr;
    model->req.override_embeds = nullptr;
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

__C void
inferBatchJiugeExWithLogits(struct JiugeModel *model,
                            const uint32_t *tokens, uint32_t ntok,
                            const uint32_t *req_lens, uint32_t nreq,
                            const uint32_t *req_pos,
                            const uint32_t *kv_pos,
                            struct KVCache **kv_caches,
                            const float *temperature, const uint32_t *topk, const float *topp,
                            uint32_t *output, void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = kv_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = 0;
    model->req.override_pos = nullptr;
    model->req.override_embeds = nullptr;
    model->req.output = output;
    model->req.logits = logits;
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
    model->req.kv_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = 0;
    model->req.override_pos = nullptr;
    model->req.override_embeds = nullptr;
    model->req.output = nullptr;
    model->req.logits = logits;
    model->req.temperature = nullptr;
    model->req.topk = nullptr;
    model->req.topp = nullptr;

    // 这是主线程（比如调用inferBatchJiuge的地方）执行的代码
    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;  // 🚦 设置绿灯信号   // 👉 拍拍工人0的肩膀："该干活了"
        lock.unlock();
        model->states[idev].cv_start.notify_one();  // 📢 喊醒对应的线程       // 📢 "醒醒！"
    }
    for (size_t i = model->dev_ids.size(); i > 0; i--) {
        auto idev = i - 1;
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
        // ⏳ 老板等待工人完成
        lock.unlock();
    }
}

__C void
forwardBatchJiugeEx(struct JiugeModel *model,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq,
                    const uint32_t *req_pos,
                    const uint32_t *kv_pos,
                    struct KVCache **kv_caches,
                    void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = kv_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = 0;
    model->req.override_pos = nullptr;
    model->req.override_embeds = nullptr;
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
inferBatchJiugeWithOverrides(struct JiugeModel *model,
                             const uint32_t *tokens, uint32_t ntok,
                             const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                             struct KVCache **kv_caches,
                             uint32_t n_override,
                             const uint32_t *override_pos,
                             const void *override_embeds,
                             const float *temperature, const uint32_t *topk, const float *topp,
                             uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = n_override;
    model->req.override_pos = override_pos;
    model->req.override_embeds = override_embeds;
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

__C void
inferBatchJiugeWithOverridesWithLogits(struct JiugeModel *model,
                                       const uint32_t *tokens, uint32_t ntok,
                                       const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                                       struct KVCache **kv_caches,
                                       uint32_t n_override,
                                       const uint32_t *override_pos,
                                       const void *override_embeds,
                                       const float *temperature, const uint32_t *topk, const float *topp,
                                       uint32_t *output, void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = n_override;
    model->req.override_pos = override_pos;
    model->req.override_embeds = override_embeds;
    model->req.output = output;
    model->req.logits = logits;
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

__C void
inferBatchJiugeWithOverridesEx(struct JiugeModel *model,
                               const uint32_t *tokens, uint32_t ntok,
                               const uint32_t *req_lens, uint32_t nreq,
                               const uint32_t *req_pos,
                               const uint32_t *kv_pos,
                               struct KVCache **kv_caches,
                               uint32_t n_override,
                               const uint32_t *override_pos,
                               const void *override_embeds,
                               const float *temperature, const uint32_t *topk, const float *topp,
                               uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = kv_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = n_override;
    model->req.override_pos = override_pos;
    model->req.override_embeds = override_embeds;
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

// __C void
// inferBatchJiugeWithOverridesExWithLogits(struct JiugeModel *model,
//                                           const uint32_t *tokens, uint32_t ntok,
//                                           const uint32_t *req_lens, uint32_t nreq,
//                                           const uint32_t *req_pos,
//                                           const uint32_t *kv_pos,
//                                           struct KVCache **kv_caches,
//                                           uint32_t n_override,
//                                           const uint32_t *override_pos,
//                                           const void *override_embeds,
//                                           const float *temperature, const uint32_t *topk, const float *topp,
//                                           uint32_t *output, void *logits) {
//     model->req.tokens = tokens;
//     model->req.ntok = ntok;
//     model->req.req_lens = req_lens;
//     model->req.nreq = nreq;
//     model->req.req_pos = req_pos;
//     model->req.kv_pos = kv_pos;
//     model->req.kv_caches = kv_caches;
//     model->req.n_override = n_override;
//     model->req.override_pos = override_pos;
//     model->req.override_embeds = override_embeds;
//     model->req.output = output;
//     model->req.logits = logits;
//     model->req.temperature = temperature;
//     model->req.topk = topk;
//     model->req.topp = topp;

//     for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
//         std::unique_lock<std::mutex> lock(model->states[idev].mtx);
//         model->states[idev].proceed = true;
//         lock.unlock();
//         model->states[idev].cv_start.notify_one();
//     }
//     for (size_t i = model->dev_ids.size(); i > 0; i--) {
//         auto idev = i - 1;
//         std::unique_lock<std::mutex> lock(model->states[idev].mtx);
//         model->states[idev].cv_done.wait(lock, [&] { return !(model->states[idev].proceed); });
//         lock.unlock();
//     }
// }

__C void
forwardBatchJiugeWithOverrides(struct JiugeModel *model,
                               const uint32_t *tokens, uint32_t ntok,
                               const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                               struct KVCache **kv_caches,
                               uint32_t n_override,
                               const uint32_t *override_pos,
                               const void *override_embeds,
                               void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = n_override;
    model->req.override_pos = override_pos;
    model->req.override_embeds = override_embeds;
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
forwardBatchJiugeWithOverridesEx(struct JiugeModel *model,
                                 const uint32_t *tokens, uint32_t ntok,
                                 const uint32_t *req_lens, uint32_t nreq,
                                 const uint32_t *req_pos,
                                 const uint32_t *kv_pos,
                                 struct KVCache **kv_caches,
                                 uint32_t n_override,
                                 const uint32_t *override_pos,
                                 const void *override_embeds,
                                 void *logits) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_pos = kv_pos;
    model->req.kv_caches = kv_caches;
    model->req.n_override = n_override;
    model->req.override_pos = override_pos;
    model->req.override_embeds = override_embeds;
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

void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, JiugeDeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    // Create Device Resource
    // 初始化设备资源
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);

    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    // 通知主线程：这个设备已经加载完成
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    // Infer Loop
    // 进入推理循环（这个线程会一直运行）
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        // 关键点：线程在这里停下来等待！
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        // quit if exit_flag is set
        if (state.exit_flag) {
            break;  // 退出线程
        }

        // 这里是关键：真正执行推理的地方！
        // 只有收到信号才会执行到这里！
        inferDeviceBatchEx(meta, *rsrc, idev, ndev, req.tokens, req.ntok,
                           req.req_lens, req.nreq, req.req_pos, req.kv_pos, req.kv_caches,
                           req.n_override, req.override_pos, req.override_embeds,
                           req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;  // 重置信号
        lock.unlock();
        // 通知主线程：这个设备完成了推理
        state.cv_done.notify_one();  // 通知主线程：我做完了
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
    dev_ids = device_ids;
    dev_resources = std::vector<JiugeDeviceResource>(ndev);  // 每个设备的资源
    states = std::vector<InferState>(ndev);                  // 每个设备的状态
    threads.resize(ndev);                                   // 每个设备的线程
    RUN_INFINI(infinirtInit());
    auto comms = std::vector<infinicclComm_t>(ndev, nullptr);
    if (ndev > 1) {
        RUN_INFINI(infinicclCommInitAll(device, comms.data(), ndev, dev_ids.data()));
    }

    // 一个卡一个线程
    for (int i = 0; i < ndev; i++) {
        // 🧵🧵🧵 这里创建线程！
        threads[i] = std::thread(launchDevice, std::cref(meta), weights, &dev_resources[i], std::ref(states[i]), std::ref(req), device, i, ndev, dev_ids[i], comms[i]);
        // ⏳ 线程立即启动，进入launchDevice函数
        // 😴 在cv_start.wait()处开始休眠等待
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
