#ifndef QWEN_MOE_IMPL_H
#define QWEN_MOE_IMPL_H

#include "infinicore_infer/models/qwen_moe.h"


// CORRECTED: Include the new shared header for common structs
#include "../common_structs.hpp" 

#include "../../allocator.hpp"
#include "../../tensor.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

// DeviceResource definition remains the same
struct DeviceResourceMoe {
    // Device
    infiniDevice_t device;
    int device_id;
    infiniopHandle_t handle;
    
    // Global Weights
    std::shared_ptr<Tensor> w_in_embd, w_out_norm, w_out_embd, sin_table, cos_table;
    
    // Attention Weights
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_qkv, b_attn_qkv, 
                                         w_attn_q_norm, w_attn_k_norm, w_attn_out;
    
    // MoE Specific Weights
    std::vector<std::shared_ptr<Tensor>> w_ffn_norm;
    std::vector<std::shared_ptr<Tensor>> w_moe_gate;
    std::vector<std::shared_ptr<Tensor>> w_moe_experts_gate_up;
    std::vector<std::shared_ptr<Tensor>> w_moe_experts_down;

    // Streams & Communicator
    infinirtStream_t stream;
    infinicclComm_t comm;

    std::shared_ptr<MemoryPool> memory_pool;
};

// NOTE: InferState, InferRequest, and KVCache have been moved to common_structs.hpp

// The main class for the MoE model instance
struct QwenMoeModel {
    QwenMoeMeta meta; 
    
    infiniDevice_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResourceMoe> dev_resources;
    std::vector<InferState> states;
    std::vector<std::thread> threads;
    InferRequest req;

    QwenMoeModel(const QwenMoeMeta *, const QwenMoeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};


#endif // QWEN_MOE_IMPL_H
