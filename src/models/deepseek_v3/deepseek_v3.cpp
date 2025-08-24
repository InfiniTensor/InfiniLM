#include "deepseek_v3_impl.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(DeepSeekV3DeviceResource *rsrc, const DeepSeekV3Meta *meta,
                          const DeepSeekV3Weights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    auto memory_pool = std::make_shared<MemoryPool>();

    *rsrc = DeepSeekV3DeviceResource{
        device,
        dev_id,
        handle,
        weights->device_weights[idev],
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(DeepSeekV3DeviceResource &res) {
    infinirtDeviceSynchronize();

    res.weights.reset();

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferDeviceBatch(const DeepSeekV3Meta &meta, DeepSeekV3DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct DeepSeekV3Cache **caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output, void *last_logits) {

    auto dt_logits = meta.dt_logits;
    // auto dt_norm = meta.dt_norm;
    // auto dt_quant_weight = meta.dt_quant_weight;
    // auto dt_quant_scale = meta.dt_quant_scale;
    // auto dt_quant_zero = meta.dt_quant_zero;
    // auto dt_gate_weight = meta.dt_gate_weight;
    // auto dt_gate_bias = meta.dt_gate_bias;
    auto n_dense_layer = meta.n_dense_layer;
    auto n_sparse_layer = meta.n_sparse_layer;
    auto nlayer = n_dense_layer + n_sparse_layer;
    size_t nh = meta.nh / size_t(ndev);

    auto d = meta.d;
    auto d_rope = meta.d_rope;
    auto d_nope = meta.d_nope;
    auto r_q = meta.r_q;
    auto r_kv = meta.r_kv;
    auto d_qk = meta.d_qk;
    auto d_v = meta.d_v;
    // auto routed_scale = meta.routed_scale;
    // auto nexperts = meta.nexperts;
    // auto kexperts = meta.kexperts;

    auto di = meta.di / size_t(ndev);
    auto dvoc = meta.dvoc;

    auto stream = rsrc.stream;

    auto weights = rsrc.weights;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

    auto q_a_buf = Tensor::buffer(dt_logits, {ntok, r_q}, rsrc.memory_pool);
    auto q_buf = Tensor::buffer(dt_logits, {ntok, nh * d_qk}, rsrc.memory_pool);
    auto kv_a_buf = Tensor::buffer(dt_logits, {ntok, r_kv + d_rope}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * d_v}, rsrc.memory_pool);

    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

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
                                       weights->w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Attention
    // attention inner
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;
    size_t max_total_len = 0;

    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
        max_total_len = std::max(max_total_len, size_t(total_len));
    }
    auto full_k_buf = Tensor::buffer(dt_logits, {max_total_len, nh * d_qk}, rsrc.memory_pool);
    auto kv_b_buf = Tensor::buffer(dt_logits, {max_total_len, nh * (d_nope + d_v)}, rsrc.memory_pool);
    auto attn_score_buf = Tensor::buffer(dt_logits, {nh, max_qk_size}, rsrc.memory_pool);
    auto attn_val_buf = Tensor::buffer(dt_logits, {nh, max_seq_len, d_v}, rsrc.memory_pool);

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        rmsnorm(logits_out, logits_in, weights->w_layers[layer].mla_norm, meta.epsilon);
        // q_proj
        dequant_linear(q_a_buf, logits_out,
                       weights->w_layers[layer].mla->q_a_proj->w,
                       weights->w_layers[layer].mla->q_a_proj->s,
                       weights->w_layers[layer].mla->q_a_proj->z,
                       1.0, 0.0, nullptr, nullptr);
        rmsnorm(q_a_buf, q_a_buf, weights->w_layers[layer].mla->q_a_norm, meta.epsilon);
        dequant_linear(q_buf, q_a_buf,
                       weights->w_layers[layer].mla->q_b_proj->w,
                       weights->w_layers[layer].mla->q_b_proj->s,
                       weights->w_layers[layer].mla->q_b_proj->z,
                       1.0, 0.0, nullptr, nullptr);
        auto q_rot = q_buf->view({ntok, nh, d_qk})->slice(2, d_nope, d_rope);
        rope(q_rot, q_rot, pos_ids_buf, weights->sin_table, weights->cos_table);
        // kv_proj
        dequant_linear(kv_a_buf, logits_out,
                       weights->w_layers[layer].mla->kv_a_proj->w,
                       weights->w_layers[layer].mla->kv_a_proj->s,
                       weights->w_layers[layer].mla->kv_a_proj->z,
                       1.0, 0.0, nullptr, nullptr);
        auto kv_pass = kv_a_buf->slice(1, 0, r_kv);
        rmsnorm(kv_pass, kv_pass, weights->w_layers[layer].mla->kv_a_norm, meta.epsilon);
        auto k_rot = kv_a_buf->slice(1, r_kv, d_rope);
        rope(k_rot, k_rot, pos_ids_buf, weights->sin_table, weights->cos_table);

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;
            auto o_req = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nh, d_v});
            auto q_req = q_buf->slice({{0, token_offset, seq_len}});
            auto kv_a_req = kv_a_buf->slice({{0, token_offset, seq_len}});
            auto kv_pass_req = kv_a_req->slice(1, 0, r_kv);
            auto k_rot_req = kv_a_req->slice(1, r_kv, d_rope);

            // concat cache
            rearrange(caches[req]->kv_pass[idev][layer]->slice(0, past_len, seq_len), kv_pass_req);
            rearrange(caches[req]->k_rot[idev][layer]->slice(0, past_len, seq_len), k_rot_req);
            // kv_b_proj
            auto kv_b_req = kv_b_buf->slice(0, 0, total_len);
            dequant_linear(kv_b_req, caches[req]->kv_pass[idev][layer]->slice(0, 0, total_len),
                           weights->w_layers[layer].mla->kv_b_proj->w,
                           weights->w_layers[layer].mla->kv_b_proj->s,
                           weights->w_layers[layer].mla->kv_b_proj->z,
                           1.0, 0.0, nullptr, nullptr);
            auto full_v_req = kv_b_req->slice(1, nh * d_qk, nh * d_v);
            // concat k
            auto full_k_req = full_k_buf->slice(0, 0, total_len);
            auto full_k_pass_req = full_k_req->slice(1, 0, nh * d_nope);
            auto full_k_rot_req = full_k_req->slice(1, nh * d_nope, nh * d_rope);
            rearrange(full_k_pass_req, kv_b_req->slice(1, 0, nh * d_nope));
            rearrange(full_k_rot_req->view({total_len, nh, d_rope}), k_rot_req->view_as({total_len, nh, d_rope}, {ptrdiff_t(d_rope), 0, 1})); // expand k_rot

            // self attention
            auto attn_score_req = attn_score_buf->slice(1, 0, seq_len * total_len)->view({nh, seq_len, total_len});
            linear(attn_score_req,
                   q_req->view({seq_len, nh, d_qk})->permute({1, 0, 2}),
                   full_k_req->view({total_len, nh, d_qk})->permute({1, 2, 0}),
                   1.f / float(sqrt(d_qk)), 0.f, nullptr, nullptr);
            // softmax
            causalSoftmax(attn_score_req, attn_score_req);
            // attn val
            auto attn_val_req = attn_val_buf->slice(1, 0, seq_len)->view({nh, seq_len, d_v});
            linear(attn_val_req, attn_score_req, full_v_req->view({total_len, nh, d_v})->permute({1, 0, 2}), 1.f, 0.f, nullptr, nullptr);
            // rearrange attn val
            rearrange(o_req, attn_val_req->permute({1, 0, 2}));

            token_offset += seq_len;
        }

        // o_proj
        dequant_linear(logits_in, o_buf,
                       weights->w_layers[layer].mla->o_proj->w,
                       weights->w_layers[layer].mla->o_proj->s,
                       weights->w_layers[layer].mla->o_proj->z,
                       1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
        // 2. MLP
        rmsnorm(logits_out, logits_in, weights->w_layers[layer].mlp_norm, meta.epsilon);

        if (layer < n_dense_layer) {
            auto gate_dense = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
            auto up_dense = Tensor::buffer(dt_logits, {ntok, di}, rsrc.memory_pool);
            dequant_linear(gate_dense, logits_out,
                           weights->w_layers[layer].dense_mlp->gate->w,
                           weights->w_layers[layer].dense_mlp->gate->s,
                           weights->w_layers[layer].dense_mlp->gate->z, 1.0, 0.0, nullptr, nullptr);
            dequant_linear(up_dense, logits_out,
                           weights->w_layers[layer].dense_mlp->up->w,
                           weights->w_layers[layer].dense_mlp->up->s,
                           weights->w_layers[layer].dense_mlp->up->z, 1.0, 0.0, nullptr, nullptr);
            swiglu(gate_dense, up_dense, gate_dense);
            dequant_linear(logits_in, gate_dense,
                           weights->w_layers[layer].dense_mlp->down->w,
                           weights->w_layers[layer].dense_mlp->down->s,
                           weights->w_layers[layer].dense_mlp->down->z,
                           1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr); // only rank 0 adds residual
        }

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
            rmsnorm(logits_out, logits_in, weights->w_out_norm, meta.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, weights->w_out_embd, 1.0, 0.0, nullptr, nullptr);
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
                        weights->w_out_norm,
                        meta.epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), weights->w_out_embd, 1.0, 0.0, nullptr, nullptr);
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
inferBatchDeepSeekV3(struct DeepSeekV3Model *model,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct DeepSeekV3Cache **kv_caches,
                     const float *temperature, const uint32_t *topk, const float *topp,
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
forwardBatchDeepSeekV3(struct DeepSeekV3Model *model,
                       const uint32_t *tokens, uint32_t ntok,
                       const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                       struct DeepSeekV3Cache **kv_caches,
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

void launchDevice(const DeepSeekV3Meta &meta, const DeepSeekV3Weights *weights, DeepSeekV3DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    CacheManager cache_manager(100);
    InferenceContext ctx(rsrc->handle, rsrc->memory_pool, &cache_manager, rsrc->stream);

    // Set the inference context for this thread
    setInferenceContext(&ctx);

    // Create Device Resource
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
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
                         req.req_lens, req.nreq, req.req_pos, req.kv_caches,
                         req.temperature, req.topk, req.topp, req.output, req.logits);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    // Clean-Up
    releaseDeviceResource(*rsrc);
    setInferenceContext(nullptr); // Clear the context when done
}

DeepSeekV3Model::DeepSeekV3Model(const DeepSeekV3Meta *_meta, const DeepSeekV3Weights *weights) : meta(*_meta) {
    int ndev = weights->device_weights.size();
    device = weights->device_weights[0]->device;
    dev_ids.resize(ndev);
    for (int i = 0; i < ndev; i++) {
        dev_ids[i] = weights->device_weights[i]->dev_id;
    }
    dev_resources = std::vector<DeepSeekV3DeviceResource>(ndev);
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

__C struct DeepSeekV3Model *
createDeepSeekV3Model(const DeepSeekV3Meta *_meta,
                      const DeepSeekV3Weights *weights) {
    DeepSeekV3Model *model = new DeepSeekV3Model(_meta, weights);
    return model;
}

__C void
destroyDeepSeekV3Model(struct DeepSeekV3Model *model) {
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
