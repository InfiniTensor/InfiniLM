#include "tinymix_impl.hpp"
#include "tinymix_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "infinicore_infer/models/tinymix.h"

#include <random>
#include <thread>
#include <vector>

void createDeviceResource(DeviceResource *rsrc, const TinyMixMeta *meta,
                          const TinyMixWeights *weights,
                          infiniDevice_t device, int idev,
                          int ndev, int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate;
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_ffn_gate_up, w_ffn_down;
    
    for (size_t layer = 0; layer < meta->nlayer; layer++) {
        w_attn_norm.push_back(
            getAttnNorm(meta, weights, layer));
        w_attn_qkv.push_back(
            getAttnQKV(meta, weights, layer, idev, ndev));
        if (weights->attn_qkv_b != nullptr) {
            b_attn_qkv.push_back(
                getAttnQKVBias(meta, weights, layer, idev, ndev));
        }
        w_attn_out.push_back(
            getAttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(
            getFFNNorm(meta, weights, layer));
        
        if (meta->nexpert > 1) {
            w_ffn_gate.push_back(getFFNGate(meta, weights, layer));
            std::vector<std::shared_ptr<Tensor>> gate_up_experts, down_experts;
            for (size_t expert = 0; expert < meta->nexpert; ++expert) {
                gate_up_experts.push_back(getFFNGateUp(meta, weights, layer, expert, idev, ndev));
                down_experts.push_back(getFFNDown(meta, weights, layer, expert, idev, ndev));
            }
            w_ffn_gate_up.push_back(gate_up_experts);
            w_ffn_down.push_back(down_experts);
        } else {
            // Placeholder for non-MoE FFN weights if needed
        }
    }

    auto memory_pool = std::make_shared<MemoryPool>(128 * 1024 * 1024);

    *rsrc = DeviceResource{
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
        w_attn_out,
        w_ffn_norm,
        w_ffn_gate,
        w_ffn_gate_up,
        w_ffn_down,
        stream,
        comm,
        memory_pool,
    };
    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(DeviceResource &res) {
    // Similar to jiuge's implementation
    infinirtDeviceSynchronize();
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();
    res.w_attn_norm.clear();
    res.w_attn_qkv.clear();
    res.b_attn_qkv.clear();
    res.w_attn_out.clear();
    res.w_ffn_norm.clear();
    res.w_ffn_gate.clear();
    res.w_ffn_gate_up.clear();
    res.w_ffn_down.clear();
    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

void inferDeviceBatch(const TinyMixMeta &meta, DeviceResource &rsrc,
                      uint32_t idev, uint32_t ndev,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output) {
    auto nlayer = meta.nlayer;
    auto nkvh = meta.nkvh / ndev;
    auto nh = meta.nh / ndev;
    auto ngroup = nh / nkvh;
    auto dh = meta.dh;
    auto d = meta.d;
    auto dt_logits = meta.dt_logits;
    auto di = meta.di / ndev;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;
    bool has_qkv_bias = rsrc.b_attn_qkv.size() > 0;

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);
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
    pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
    RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(), sizeof(uint32_t) * ntok,
                                    INFINIRT_MEMCPY_H2D, stream));

    for (uint32_t i = 0; i < ntok; i++) {
        RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                       rsrc.w_in_embd->data(tokens[i] * d),
                                       dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
    }

    // Prepare operators and workspace
    size_t workspace_size = 0;
    // ... (Attention descriptor creation logic will be added here later) ...

    // For now, let's focus on the FFN descriptors.
    // We create them once and reuse them in the loop.
    infiniopRMSNormDescriptor_t desc_norm_ffn;
    RUN_INFINI(infiniopCreateRMSNormDescriptor(rsrc.handle, &desc_norm_ffn, logits_in->desc(), logits_out->desc(), rsrc.w_ffn_norm[0]->desc(), meta.epsilon));
    
    // Descriptors for standard FFN path
    auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di}, rsrc.memory_pool);
    auto gate_buf = gate_up_buf->slice(1, 0, di);
    auto up_buf = gate_up_buf->slice(1, di, di);

    infiniopGemmDescriptor_t desc_ffn_gate_up, desc_ffn_down;
    infiniopSwiGLUDescriptor_t desc_swiglu;

    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_ffn_gate_up, gate_up_buf->desc(), logits_out->desc(), rsrc.w_ffn_gate_up[0][0]->desc()));
    RUN_INFINI(infiniopCreateSwiGLUDescriptor(rsrc.handle, &desc_swiglu, gate_buf->desc(), up_buf->desc(), gate_buf->desc()));
    RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_ffn_down, logits_in->desc(), gate_buf->desc(), rsrc.w_ffn_down[0][0]->desc()));

    // ... (Workspace size calculation for all operators) ...
    // Allocate a single workspace for all ops.
    // void* workspace = ...;

    // Compute
    for (uint32_t layer = 0; layer < nlayer; layer++) {
        // 1. Attention logic would be here...

        // 2. FFN / MoE
        RUN_INFINI(infiniopRMSNorm(desc_norm_ffn, nullptr, 0, logits_out->data(), logits_in->data(), rsrc.w_ffn_norm[layer]->data(), stream));
        
        if (meta.nexpert > 1) {
            // MOE LOGIC
            auto gating_scores = Tensor::buffer(dt_logits, {ntok, meta.nexpert}, rsrc.memory_pool);
            
            // Step 1: Gating GEMM
            infiniopGemmDescriptor_t desc_gating_gemm;
            RUN_INFINI(infiniopCreateGemmDescriptor(rsrc.handle, &desc_gating_gemm, gating_scores->desc(), logits_out->desc(), rsrc.w_ffn_gate[layer]->desc()));
            RUN_INFINI(infiniopGemm(desc_gating_gemm, nullptr, 0, gating_scores->data(), logits_out->data(), rsrc.w_ffn_gate[layer]->data(), 1.0, 0.0, stream));
            // Remember to destroy the descriptor after use
            infiniopDestroyGemmDescriptor(desc_gating_gemm);

            // Step 2: Softmax is fused in our TopK operator, so we call TopK directly.
            auto topk_val = Tensor::buffer(dt_logits, {ntok, meta.topk}, rsrc.memory_pool);
            auto topk_ind = Tensor::buffer(INFINI_DTYPE_I32, {ntok, meta.topk}, rsrc.memory_pool);

            infiniopTopKDescriptor_t desc_topk;
            RUN_INFINI(infiniopCreateTopKDescriptor(rsrc.handle, &desc_topk, gating_scores->desc(), topk_val->desc(), topk_ind->desc(), meta.topk));
            
            size_t workspace_size;
            RUN_INFINI(infiniopGetTopKWorkspaceSize(desc_topk, &workspace_size));
            auto workspace = Tensor::buffer(INFINI_DTYPE_U8, {workspace_size}, rsrc.memory_pool);

            RUN_INFINI(infiniopTopK(desc_topk, workspace->data(), workspace_size, topk_val->data(), topk_ind->data(), gating_scores->data(), stream));
            
            infiniopDestroyTopKDescriptor(desc_topk);

            // TODO: Step 3: Dispatch & Combine...
        } else {
            // Standard FFN Logic
            RUN_INFINI(infiniopGemm(desc_ffn_gate_up, nullptr, 0, gate_up_buf->data(), logits_out->data(), rsrc.w_ffn_gate_up[layer][0]->data(), 1.0, 0.0, stream));
            RUN_INFINI(infiniopSwiGLU(desc_swiglu, nullptr, 0, gate_buf->data(), up_buf->data(), gate_buf->data(), stream));
            RUN_INFINI(infiniopGemm(desc_ffn_down, nullptr, 0, logits_in->data(), gate_buf->data(), rsrc.w_ffn_down[layer][0]->data(), 1.0, 1.0, stream)); // Add residual
        }
        
        // ... (AllReduce) ...
    }

    // ... (Sampling logic) ...

    // Clean up descriptors
    infiniopDestroyRMSNormDescriptor(desc_norm_ffn);
    infiniopDestroyGemmDescriptor(desc_ffn_gate_up);
    infiniopDestroySwiGLUDescriptor(desc_swiglu);
    infiniopDestroyGemmDescriptor(desc_ffn_down);
    // ... (Destroy other descriptors) ...
}

// Boilerplate code for model creation, destruction, and thread management
// This part is very similar to jiuge.cpp and can be adapted directly.
// To save space, I will omit the full copy-paste here but it should include:
// - launchDevice
// - TinyMixModel::TinyMixModel
// - createTinyMixModel
// - destroyTinyMixModel
// - inferBatchTinyMix

void launchDevice(const TinyMixMeta &meta, const TinyMixWeights *weights, DeviceResource *rsrc, InferState &state, InferRequest &req,
                  infiniDevice_t device, int idev, int ndev, int dev_id, infinicclComm_t comm) {
    createDeviceResource(rsrc, &meta, weights, device, idev, ndev, dev_id, comm);
    {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.loaded = true;
        lock.unlock();
        state.cv_load.notify_one();
    }

    while (true) {
        std::unique_lock<std::mutex> lock(state.mtx);
        state.cv_start.wait(lock, [&] { return state.proceed || state.exit_flag; });
        if (state.exit_flag) {
            break;
        }

        inferDeviceBatch(meta, *rsrc, idev, ndev, req.tokens, req.ntok, req.req_lens, req.nreq, req.req_pos, req.kv_caches, req.temperature, req.topk, req.topp, req.output);

        state.proceed = false;
        lock.unlock();
        state.cv_done.notify_one();
    }

    releaseDeviceResource(*rsrc);
}

TinyMixModel::TinyMixModel(const TinyMixMeta *_meta, const TinyMixWeights *weights, infiniDevice_t device_, std::vector<int> device_ids) : meta(*_meta) {
    int ndev = int(device_ids.size());
    device = device_;
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
    for (int i = 0; i < ndev; i++) {
        std::unique_lock<std::mutex> lock(states[i].mtx);
        states[i].cv_load.wait(lock, [&] { return states[i].loaded; });
        lock.unlock();
    }
}

__C struct TinyMixModel *
createTinyMixModel(const TinyMixMeta *meta,
                 const TinyMixWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids) {
    std::vector<int> device_ids(ndev);
    std::copy(dev_ids, dev_ids + ndev, device_ids.begin());
    TinyMixModel *model = new TinyMixModel(meta, weights, device, device_ids);
    return model;
}

__C void destroyTinyMixModel(struct TinyMixModel *model) {
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

__C void
inferBatchTinyMix(struct TinyMixModel *model,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output) {
    model->req.tokens = tokens;
    model->req.ntok = ntok;
    model->req.req_lens = req_lens;
    model->req.nreq = nreq;
    model->req.req_pos = req_pos;
    model->req.kv_caches = kv_caches;
    model->req.output = output;
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
