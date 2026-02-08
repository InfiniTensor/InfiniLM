#ifndef QWEN3MOE_IMPL_H
#define QWEN3MOE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct QuantLinearWeight {
    std::shared_ptr<Tensor> w;
    std::shared_ptr<Tensor> s; // Scale QUANT
    std::shared_ptr<Tensor> z; // Zero QUANT
}; 

struct Qwen3AttentionWeight {
    // Pre-Norm
    std::shared_ptr<Tensor> attn_norm;
    
    // GQA
    std::shared_ptr<Tensor> q_proj;
    std::shared_ptr<Tensor> k_proj;
    std::shared_ptr<Tensor> v_proj;
    std::shared_ptr<Tensor> o_proj;

    // QK Norm
    std::shared_ptr<Tensor> q_norm;
    std::shared_ptr<Tensor> k_norm;

};

struct Qwen3LayerWeight {
    std::shared_ptr<Qwen3AttentionWeight> self_attn;

    // TODO: 实现MLP Experts等， 由于比赛只实现attention模块
    // 所以只放一个self_attn
};

struct Qwen3DeviceWeights {
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd;
    
    // RoPE
    std::shared_ptr<Tensor> sin_table;
    std::shared_ptr<Tensor> cos_table;

    // layer
    std::vector<Qwen3LayerWeight> w_layers;
    
    infiniDevice_t device;
    int dev_id;
    infinirtStream_t load_stream;
};

struct Qwen3MoEWeights {
    // 即使是单卡，通常也用 vector 存，方便统一逻辑
    std::vector<std::shared_ptr<Qwen3DeviceWeights>> device_weights;

    // 构造函数声明
    Qwen3MoEWeights(const Qwen3MoEAttentionMeta *meta,
                    infiniDevice_t device,
                    int ndev,
                    const int *dev_ids);
};

/*
Qwen3 KVCache
[Batch, KV_Heads, Max_Seq, Head_Dim]    
*/
struct Qwen3Cache {
    std::vector<std::pair<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>> layers;
};

struct Qwen3MoEDeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Qwen3DeviceWeights> weights;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};


struct InferState {
    std::mutex mtx;
    std::condition_variable cv_load, cv_start, cv_done;
    bool loaded = false;
    bool proceed = false;
    bool exit_flag = false;
};

struct InferRequest {
    const uint32_t *tokens;
    uint32_t ntok;
    const uint32_t *req_lens;
    uint32_t nreq;
    const uint32_t *req_pos;
    struct Qwen3Cache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct Qwen3MoEAttention { 
    Qwen3MoEAttentionMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    
    std::vector<Qwen3MoEDeviceResource> dev_resources;
    
    // 线程控制
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    // 构造函数
    Qwen3MoEAttention(const Qwen3MoEAttentionMeta *, const Qwen3MoEWeights *weights);
};


#endif