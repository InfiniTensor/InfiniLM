#ifndef QWEN3VL_IMPL_H
#define QWEN3VL_IMPL_H

#include "infinicore_infer.h"

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

struct LayerWeight {
    std::shared_ptr<Tensor> attn_norm;
    std::shared_ptr<Tensor> attn_qkv_proj;
    std::shared_ptr<Tensor> attn_q_norm;
    std::shared_ptr<Tensor> attn_k_norm;
    std::shared_ptr<Tensor> attn_o_proj;

    std::shared_ptr<Tensor> mlp_norm;
    std::shared_ptr<Tensor> mlp_down, mlp_gate_up;
};

struct LanguageModelWeight {
    std::shared_ptr<Tensor> in_embd, out_embd, out_norm;
    std::vector<LayerWeight> layers;
};

struct VisBlockWeight {
    std::shared_ptr<Tensor> attn_proj_weight, attn_proj_bias, attn_qkv_weight, attn_qkv_bias;
    std::shared_ptr<Tensor> mlp_linear_fc1_weight, mlp_linear_fc1_bias, mlp_linear_fc2_weight, mlp_linear_fc2_bias;
    std::shared_ptr<Tensor> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
};

struct DeepstackMergerWeight {
    std::shared_ptr<Tensor> linear_fc1_weight, linear_fc1_bias, linear_fc2_weight, linear_fc2_bias;
    std::shared_ptr<Tensor> norm_weight, norm_bias;
};

struct MergerWeight {
    std::shared_ptr<Tensor> linear_fc1_weight, linear_fc1_bias, linear_fc2_weight, linear_fc2_bias;
    std::shared_ptr<Tensor> norm_weight, norm_bias;
};


struct VisualEncoderWeight {
    std::shared_ptr<Tensor> patch_embed_weight, patch_embed_bias, pos_embed_weight;
    std::vector<VisBlockWeight> blocks;
    std::vector<DeepstackMergerWeight> deepstack_mergers;
    std::shared_ptr<MergerWeight> merger;
};


struct Qwen3vlDeviceWeights {
    std::shared_ptr<Tensor> sin_table,cos_table;
    std::shared_ptr<LanguageModelWeight> w_lang;
    std::shared_ptr<VisualEncoderWeight> w_vis;
    infiniDevice_t device;
    int dev_id;
    infinirtStream_t load_stream;
};

struct Qwen3vlWeights {
    Qwen3vlMeta const *meta;
    bool transpose_weight;
    std::vector<std::shared_ptr<Qwen3vlDeviceWeights>> device_weights;

    Qwen3vlWeights(const Qwen3vlMeta *meta,
                      infiniDevice_t device,
                      int ndev,
                      const int *dev_ids,
                      bool transpose_weight);
};

struct Qwen3vlDeviceResource {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Qwen3vlDeviceWeights> weights;
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
    struct Qwen3vlCache **kv_caches;
    const float *temperature;
    const uint32_t *topk;
    const float *topp;
    uint32_t *output;
    void *logits;
};

struct Qwen3vlModel {
    Qwen3vlMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<Qwen3vlDeviceResource> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    Qwen3vlModel(const Qwen3vlMeta *, const Qwen3vlWeights *weights);
};

struct Qwen3vlCache {
    std::vector<std::vector<std::shared_ptr<Tensor>>> k_rot, v; 
};

#endif