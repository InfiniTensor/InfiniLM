#ifndef JIUGE_IMPL_H
#define JIUGE_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct DeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    bool is_quantized = false;
    bool symmetric = true;
    int bits = 8;
    int group_size = 128;
    std::vector<std::shared_ptr<Tensor>> w_attn_qkv_qweight, w_attn_qkv_scales, w_attn_qkv_qzeros, w_attn_qkv_g_idx;
    std::vector<std::shared_ptr<Tensor>> w_attn_o_qweight, w_attn_o_scales, w_attn_o_qzeros, w_attn_o_g_idx;
    std::vector<std::shared_ptr<Tensor>> w_ffn_gate_up_qweight, w_ffn_gate_up_scales, w_ffn_gate_up_qzeros, w_ffn_gate_up_g_idx;
    std::vector<std::shared_ptr<Tensor>> w_ffn_down_qweight, w_ffn_down_scales, w_ffn_down_qzeros, w_ffn_down_g_idx;

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
    struct KVCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct JiugeModel {
    JiugeMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    JiugeModel(const JiugeMeta *, const JiugeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

struct KVCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k, v;
};

#endif
