#ifndef LLAVA_IMPL_HPP
#define LLAVA_IMPL_HPP

#include "infinicore_infer/models/llava.h"
#include "../../allocator.hpp"
#include "../../tensor.hpp"
#include "../../cache.hpp"  // 添加KV Cache支持

#include <mutex>
#include <condition_variable>
#include <thread>
#include <vector>



// 设备资源结构 - 统一线程架构只需要一套resource
struct LlavaDeviceResource {
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    // Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table,
        cos_table;
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, w_attn_q_norm, w_attn_k_norm,w_attn_out,
        w_ffn_norm, w_ffn_gate_up, w_ffn_down;
    // Streams
    infinirtStream_t stream;
    // Communicator
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

// 最简单的推理状态结构
struct LlavaInferState {
    std::mutex mtx;
    std::condition_variable cv_stage;     // 使用cv_stage进行同步

    bool proceed = false;
    bool exit_flag = false;
    int current_stage = 0;
    bool stage_completed = false;
    bool error_occurred = false;
    std::string error_message;

    // 添加loaded标志，与jiuge.cpp保持一致
    bool loaded = false;

    const uint32_t* input_tokens = nullptr;
    const void* image_data = nullptr;
    void** kv_caches = nullptr;
    uint32_t ntok = 0;
    uint32_t nreq = 0;
    uint32_t batch_size = 0;
    uint32_t* output = nullptr;
    void* vision_features = nullptr;
    void* projected_features = nullptr;
};

// // 推理请求结构
// struct LlavaRequest {
//     const uint32_t* input_tokens;
//     const void* image_data;
//     void** kv_caches;
//     uint32_t ntok;
//     uint32_t nreq;
//     uint32_t batch_size;
//     uint32_t* output;
// } req;

// TODO: 想想这里需要啥
struct LlavaRequest {
    const uint32_t* input_tokens;
    const void* image_data;
    void** kv_caches;
    uint32_t ntok;
    uint32_t nreq;
    uint32_t batch_size;
    uint32_t* output;
};

struct LlavaModel {
    LlavaMeta meta;
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<LlavaDeviceResource> dev_resources;
    std::vector<LlavaInferState> states;
    std::vector<std::thread> threads;  // 添加线程向量
    LlavaRequest req;

    LlavaModel(const LlavaMeta *, const LlavaWeights *,
               infiniDevice_t device, std::vector<int> device_ids);
    // ~LlavaModel();

    // // 统一推理接口
    // void inferBatchLlava(const uint32_t* input_tokens, const void* image_data,
    //                     void** kv_caches, const char* mode, uint32_t batch_size,
    //                     uint32_t* output);

};

#endif