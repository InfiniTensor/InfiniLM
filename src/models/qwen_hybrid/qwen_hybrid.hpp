#pragma once
#include "infinicore_infer/models/qwen_hybrid.h"

#include "../../cache.hpp"
#include "../../dataloader/weights_loader.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

struct QwenHybridDeviceWeight {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;

    std::vector<std::shared_ptr<Tensor>> w_attn_norm;
    // full attention
    std::vector<std::shared_ptr<Tensor>> b_attn_q, b_attn_k, b_attn_v;
    std::vector<std::shared_ptr<Tensor>> w_attn_q, w_attn_k, w_attn_v, w_attn_out;
    // linear attention
    std::vector<std::shared_ptr<Tensor>> b_la_dt, alpha_la_g, w_la_conv, w_la_qkvz, w_la_ba, w_la_norm, w_la_out;
    
    std::vector<std::shared_ptr<Tensor>> w_ffn_norm;
    // ----------------------------------------------------------------------- //
    //                                       moe                               //
    // ----------------------------------------------------------------------- //
    std::vector<std::shared_ptr<Tensor>> w_shared_expert_gate; // gata 权重
    std::vector<std::shared_ptr<Tensor>> w_router_expert_gate;

    std::vector<std::shared_ptr<Tensor>> w_shared_expert_ffn_gate; // 共享专家的权重
    std::vector<std::shared_ptr<Tensor>> w_shared_expert_ffn_up;
    std::vector<std::shared_ptr<Tensor>> w_shared_expert_ffn_down;

    std::vector<std::vector<std::shared_ptr<Tensor>>> w_router_expert_ffn_gate; // 路由专家的权重
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_router_expert_ffn_up;
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_router_expert_ffn_down;
};

class QwenHybridWeights : public infinicore::weights::Loader {
private:
    std::vector<std::shared_ptr<QwenHybridDeviceWeight>> _device_weights;

public:
    QwenHybridWeights(const QwenHybridMeta *meta,
                      infiniDevice_t device,
                      const std::vector<int> &dev_ids);
    std::vector<std::shared_ptr<QwenHybridDeviceWeight>> &device_weights() {
        return _device_weights;
    }
};

struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<QwenHybridDeviceWeight> weights;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    KVCache **kv_caches;
    MambaCache **mamba_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct QwenHybridModel {
    QwenHybridMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    QwenHybridModel(const QwenHybridMeta *, const ModelWeights *);
};