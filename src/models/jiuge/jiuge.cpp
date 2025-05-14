#include "jiuge_impl.hpp"
#include "jiuge_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer.h"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(DeviceResource *rsrc, const JiugeMeta *meta,
                          const JiugeWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            get_attn_norm(meta, weights, layer));
        w_attn_qkv.push_back(
            get_attn_qkv(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                get_attn_qkv_bias(meta, weights, layer, idev, ndev));
        }

        w_attn_out.push_back(
            get_attn_o(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            get_ffn_norm(meta, weights, layer));
        w_ffn_gate_up.push_back(
            get_ffn_gate_up(meta, weights, layer, idev, ndev));
        w_ffn_down.push_back(
            get_ffn_down(meta, weights, layer, idev, ndev));
    }

    *rsrc = DeviceResource{device,
                           dev_id,
                           handle,
                           get_in_embd(meta, weights),
                           get_out_norm(meta, weights),
                           get_out_embd(meta, weights),
                           get_sin_table(meta),
                           get_cos_table(meta),
                           w_attn_norm,
                           w_attn_qkv,
                           b_attn_qkv,
                           w_attn_out,
                           w_ffn_norm,
                           w_ffn_gate_up,
                           w_ffn_down,
                           stream,
                           comm};
}

void inferDeviceBatch(const JiugeMeta &meta, const DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      uint32_t *ans,
                      float temperature, uint32_t topk, float topp) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    // auto dctx = meta.dctx;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, stream);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, stream);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, stream);
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, stream);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, stream);
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, stream);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_U32, {nreq}, stream);
    auto result_cpu = std::vector<uint32_t>(nreq);
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
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, stream);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                       INFINIRT_MEMCPY_H2D, stream));
    }
    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Prepare operators and workspace
    void *workspace;
    size_t workspace_size = 0, temp_size = 0;
    // attn & mlp rmsnorm
    infiniopRMSNormDescriptor_t desc_norm;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm, logits_in->desc()->get(),
        logits_out->desc()->get(), rsrc.w_attn_norm[0]->desc()->get(),
        meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm, &workspace_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Attention
    infiniopGemmDescriptor_t desc_attn_qkv, desc_attn_o;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_qkv, qkv_buf->desc()->get(),
        logits_in->desc()->get(), rsrc.w_attn_qkv[0]->desc()->get()));
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_attn_o, logits_in->desc()->get(),
        o_buf->desc()->get(), rsrc.w_attn_out[0]->desc()->get()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_qkv, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_attn_o, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRoPEDescriptor_t desc_rope_q, desc_rope_k;
    qkv_buf->dim_split(1, {nh + nkvh * 2, dh}); // (ntok, nh + 2 * nkvh, dh)
    auto qkv_buf_q = qkv_buf->slice(1, 0, nh);
    auto qkv_buf_k = qkv_buf->slice(1, nh, nkvh);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_q, qkv_buf_q->desc()->get(), qkv_buf_q->desc()->get(),
        pos_ids_buf->desc()->get(), rsrc.sin_table->desc()->get(),
        rsrc.cos_table->desc()->get()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_q, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateRoPEDescriptor(
        rsrc.handle, &desc_rope_k, qkv_buf_k->desc()->get(), qkv_buf_k->desc()->get(),
        pos_ids_buf->desc()->get(), rsrc.sin_table->desc()->get(),
        rsrc.cos_table->desc()->get()));
    RUN_INFINI(infiniopGetRoPEWorkspaceSize(desc_rope_k, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // attention inner
    auto desc_attns = std::vector<infiniopAttentionDescriptor_t>(nreq);
    size_t token_offset = 0;
    o_buf->dim_split(1, {nh, dh});
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto o = o_buf->slice({{0, token_offset, seq_len}});
        auto q = qkv_buf->slice({{0, token_offset, seq_len}, {1, 0, nh}})
                     ->permute({1, 0, 2});
        auto k = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh, nkvh}})
                     ->permute({1, 0, 2});
        auto v = qkv_buf->slice({{0, token_offset, seq_len}, {1, nh + nkvh, nkvh}})
                     ->permute({1, 0, 2});
        auto k_cache = kv_caches[req]->k[idev][0];
        auto v_cache = kv_caches[req]->v[idev][0];
        RUN_INFINI(infiniopCreateAttentionDescriptor(
            rsrc.handle, &desc_attns[req], o->desc()->get(), q->desc()->get(),
            k->desc()->get(), v->desc()->get(), k_cache->desc()->get(),
            v_cache->desc()->get(), past_len));
        RUN_INFINI(
            infiniopGetAttentionWorkspaceSize(desc_attns[req], &temp_size));
        workspace_size = std::max(workspace_size, temp_size);
        token_offset += seq_len;
    }

    // MLP descriptors
    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc()->get(),
        logits_out->desc()->get(), rsrc.w_ffn_gate_up[0]->desc()->get()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_gate_up, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(
        rsrc.handle, &desc_swiglu, logits_out->desc()->get(), up_buf->desc()->get(), gate_buf->desc()->get()));
    RUN_INFINI(infiniopGetSwiGLUWorkspaceSize(desc_swiglu, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_ffn_down, logits_in->desc()->get(),
        logits_out->desc()->get(), rsrc.w_ffn_down[0]->desc()->get()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_ffn_down, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);

    // Output and sample
    infiniopRMSNormDescriptor_t desc_norm_out;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(
        rsrc.handle, &desc_norm_out, logits_out->slice(0, 0, 1)->desc()->get(),
        logits_out->slice(0, 0, 1)->desc()->get(),
        rsrc.w_out_norm->desc()->get(), meta.epsilon));
    RUN_INFINI(infiniopGetRMSNormWorkspaceSize(desc_norm_out, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopGemmDescriptor_t desc_out_embd;
    RUN_INFINI(infiniopCreateGemmDescriptor(
        rsrc.handle, &desc_out_embd, prob_buf->desc()->get(),
        logits_out->slice(0, 0, nreq)->desc()->get(),
        rsrc.w_out_embd->desc()->get()));
    RUN_INFINI(infiniopGetGemmWorkspaceSize(desc_out_embd, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    infiniopRandomSampleDescriptor_t desc_sample;
    RUN_INFINI(infiniopCreateRandomSampleDescriptor(
        rsrc.handle, &desc_sample,
        TensorDesc::create(INFINI_DTYPE_U64, {1}, {1})->get(),
        TensorDesc::create(dt_logits, {dvoc}, {1})->get()));
    RUN_INFINI(infiniopGetRandomSampleWorkspaceSize(desc_sample, &temp_size));
    workspace_size = std::max(workspace_size, temp_size);
    // Allocate workspace
    RUN_INFINI(infinirtMallocAsync(&workspace, workspace_size, stream));
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention
        // rms norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_attn_norm[layer]->data(), stream));
        // qkv_proj
        RUN_INFINI(infiniopGemm(
            desc_attn_qkv, workspace, workspace_size,
            qkv_buf->data(), logits_out->data(),
            rsrc.w_attn_qkv[layer]->data(), 1.0, 0.0, stream));
        // rope
        RUN_INFINI(infiniopRoPE(
            desc_rope_q, workspace, workspace_size,
            qkv_buf->data(), qkv_buf->data(),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(), stream));
        RUN_INFINI(infiniopRoPE(
            desc_rope_k, workspace, workspace_size,
            qkv_buf->data(nh * dh), qkv_buf->data(nh * dh),
            pos_ids_buf->data(),
            rsrc.sin_table->data(),
            rsrc.cos_table->data(),
            stream));

        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            // self attention
            RUN_INFINI(infiniopAttention(
                desc_attns[req], workspace, workspace_size,
                o_buf->data(token_offset * nh * dh),
                qkv_buf->data(token_offset * (nh + nkvh * 2) * dh),
                qkv_buf->data(token_offset * (nh + nkvh * 2) * dh + nh * dh),
                qkv_buf->data(token_offset * (nh + nkvh * 2) * dh + (nh + nkvh) * dh),
                kv_caches[req]->k[idev][layer]->data(),
                kv_caches[req]->v[idev][layer]->data(),
                stream));

            token_offset += seq_len;
        }
        // o_proj
        RUN_INFINI(infiniopGemm(
            desc_attn_o, workspace, workspace_size,
            logits_in->data(), o_buf->data(),
            rsrc.w_attn_out[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
        }

        // 2. FFN
        // rms_norm
        RUN_INFINI(infiniopRMSNorm(
            desc_norm, workspace, workspace_size,
            logits_out->data(), logits_in->data(),
            rsrc.w_ffn_norm[layer]->data(), stream));
        // mlp
        RUN_INFINI(infiniopGemm(
            desc_ffn_gate_up, workspace, workspace_size,
            gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer]->data(),
            1.0, 0.0, stream));
        RUN_INFINI(infiniopSwiGLU(
            desc_swiglu, workspace, workspace_size,
            logits_out->data(), up_buf->data(), gate_buf->data(), stream));
        RUN_INFINI(infiniopGemm(
            desc_ffn_down, workspace, workspace_size,
            logits_in->data(), logits_out->data(),
            rsrc.w_ffn_down[layer]->data(), 1.0, idev == 0 ? 1.0 : 0.0, stream)); // only rank 0 adds residual

        // All_reduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(
                logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                INFINICCL_SUM, rsrc.comm, stream));
        }
    }
    // Sample and Output
    uint64_t tmp;
    if (idev == 0) {
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            token_offset += seq_len;
            RUN_INFINI(infiniopRMSNorm(
                desc_norm_out, workspace, workspace_size,
                logits_out->data(req * d),
                logits_in->data((token_offset - 1) * d),
                rsrc.w_out_norm->data(), stream));
        }
        RUN_INFINI(infiniopGemm(
            desc_out_embd, workspace, workspace_size,
            prob_buf->data(), logits_out->data(),
            rsrc.w_out_embd->data(), 1.0, 0.0, stream));
        std::random_device _rd;
        std::mt19937 gen(_rd());
        token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto seq_len = req_lens[req];
            float random_val = std::uniform_real_distribution<float>(0, 1)(gen);
            RUN_INFINI(infiniopRandomSample(
                desc_sample, workspace, workspace_size,
                result_buf->data(req),
                prob_buf->data(req * dvoc), random_val, topp,
                topk, temperature, stream));
            token_offset += seq_len;
        }
        RUN_INFINI(infinirtStreamSynchronize(stream));
        RUN_INFINI(infinirtMemcpy(&tmp, result_buf->data(),
                                  sizeof(uint64_t) * nreq, INFINIRT_MEMCPY_D2H));
        for (uint32_t req = 0; req < nreq; req++) {
            ans[req] = (uint32_t)result_cpu[req];
        }
    }

    // Clean up
    infiniopDestroyRMSNormDescriptor(desc_norm);
    infiniopDestroyGemmDescriptor(desc_attn_qkv);
    infiniopDestroyGemmDescriptor(desc_attn_o);
    infiniopDestroyRoPEDescriptor(desc_rope_q);
    infiniopDestroyRoPEDescriptor(desc_rope_k);
    for (uint32_t req = 0; req < nreq; req++) {
        infiniopDestroyAttentionDescriptor(desc_attns[req]);
    }
    infiniopDestroyRMSNormDescriptor(desc_norm_out);
    infiniopDestroyGemmDescriptor(desc_out_embd);
    infiniopDestroyRandomSampleDescriptor(desc_sample);
    infinirtFree(workspace);
}

__C void
inferBatch(struct JiugeModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           uint32_t *ans,
           float temperature, uint32_t topk, float topp) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.ans = ans;
    model->req.temperature = temperature;
    model->req.topk = topk;
    model->req.topp = topp;

    for (size_t idev = 0; idev < model->dev_ids.size(); idev++) {
        std::unique_lock<std::mutex> lock(model->states[idev].mtx);
        model->states[idev].proceed = true;
        lock.unlock();
        model->states[idev].cv.notify_one();
    }
}

void launchDevice(const JiugeMeta &meta, const JiugeWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv.wait(lock, [&] { return state.proceed || state.exit_flag; });
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.ans, req.temperature, req.topk, req.topp);

        state.proceed = false;
        lock.unlock();
    }

    infiniopDestroyHandle(rsrc->handle);
    infinirtStreamDestroy(rsrc->stream);
    infinicclCommDestroy(rsrc->comm);
}

JiugeModel::JiugeModel(const JiugeMeta *_meta, const JiugeWeights *weights, infiniDevice_t device, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    dev_ids = device_ids;
    dev_resources = std::vector<DeviceResource>(ndev);
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
        model->states[idev].cv.notify_one();
    }

    for (size_t idev = 0; idev < ndev; idev++) {
        model->threads[idev].join();
    }

    delete model;
}