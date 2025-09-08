#pragma once
#include "infinicore_infer/models/qwen_hybrid.h"

#include "../../dataloader/weights_loader.hpp"

#include <condition_variable>
#include <mutex>
#include <thread>

struct QwenHybridCache {
};

struct QwenHybridDeviceWeight {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, b_attn_q, b_attn_k, b_attn_v, w_ffn_norm;
    std::vector<std::shared_ptr<Tensor>> w_attn_q, w_attn_k, w_attn_v, w_attn_out, w_ffn_gate, w_ffn_up, w_ffn_down;
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
    QwenHybridCache **caches;
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