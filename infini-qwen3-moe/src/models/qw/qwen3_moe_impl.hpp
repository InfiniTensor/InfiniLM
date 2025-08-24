#pragma once

#include <infinicore_infer/models/qwen3_moe.h>
#include <tensor.hpp>
#include <memory>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

/*
 * Qwen3-MoE模型实现
 * 
 * 基于Qwen3架构，增加了Mixture of Experts (MoE)支持：
 * - 稀疏激活的专家网络
 * - 路由器网络进行专家选择
 * - 分布式推理中的专家权重分区
 * - 负载均衡和辅助损失计算
 * 
 * MoE层结构：
 * - Router: 选择top-k个专家
 * - Experts: 多个独立的MLP专家
 * - Combining: 根据路由权重组合专家输出
 * 
 * 分布式策略：
 * - 专家权重在设备间分区
 * - 注意力权重遵循原有分区方案
 * - 路由计算在每个设备上复制
 */

/*
 * 推理请求结构
 * 
 * 包含单次批量推理所需的所有输入数据。
 * 在分布式设置中在所有设备间共享以协调并行推理。
 */
struct Qwen3MoeInferRequest {
    // ================================
    // Input Token Data
    // ================================
    const uint32_t *tokens;               // Input token sequence [ntok]
    uint32_t ntok;                        // Total number of input tokens
    
    // ================================
    // Request Batching Information
    // ================================
    const uint32_t *req_lens;             // Length of each request [nreq]
    uint32_t nreq;                        // Number of requests in the batch
    const uint32_t *req_pos;              // Starting position for each request [nreq]
    
    // ================================
    // KV Cache Management
    // ================================
    struct Qwen3MoeKVCache **kv_caches;   // KV cache for each request [nreq]
    
    // ================================
    // Sampling Parameters
    // ================================
    const float *temperature;             // Sampling temperature [nreq]
    const uint32_t *topk;                 // Top-k sampling parameter [nreq]
    const float *topp;                    // Top-p (nucleus) sampling parameter [nreq]
    
    // ================================
    // Output Storage
    // ================================
    uint32_t *output;                     // Output tokens [nreq]
};

/*
 * 推理状态结构
 * 
 * 管理设备工作线程的线程同步和生命周期。
 * 每个设备都有一个关联的 InferState 用于协调异步推理。
 * 
 * 线程同步模式：
 * 1. 主线程在 cv_load 上等待直到 loaded=true（设备初始化完成）
 * 2. 主线程设置 proceed=true 并通知 cv_start 触发推理
 * 3. 工作线程处理推理并设置 proceed=false，通知 cv_done
 * 4. 主线程在 cv_done 上等待直到 proceed=false（推理完成）
 */
struct Qwen3MoeInferState {
    // ================================
    // Synchronization Primitives
    // ================================
    std::mutex mtx;                       // Mutex for state protection
    std::condition_variable cv_load;      // Condition var for device loading
    std::condition_variable cv_start;     // Condition var for inference start
    std::condition_variable cv_done;      // Condition var for inference completion
    
    // ================================
    // State Flags
    // ================================
    volatile bool loaded;                 // Device initialization complete flag
    volatile bool proceed;                // Inference execution flag
    volatile bool shutdown;               // Thread shutdown signal
};

/*
 * 设备特定的Qwen3-MoE资源
 * 
 * 每个设备维护自己的权重分区和计算资源：
 * - 注意力权重：按头维度在设备间分区
 * - MLP权重：按中间维度在设备间分区  
 * - MoE专家权重：专家在设备间分区，每个设备处理部分专家
 * - KV 缓存：按注意力头在设备间分区
 */
struct DeviceQwen3MoeResource {
    // ================================
    // Device Context and Handles
    // ================================
    infiniDevice_t device;               // InfiniCore device type (GPU/CPU)
    int device_id;                       // Physical device ID for this resource
    infiniopHandle_t handle;             // InfiniCore operation handle for compute ops
    infinirtStream_t stream;             // Execution stream for asynchronous operations
    infinicclComm_t comm;               // Inter-device communicator for distributed ops
    
    // ================================
    // Memory Management
    // ================================
    std::shared_ptr<MemoryPool> memory_pool;  // Pool for temporary buffer allocation
    
    // ================================
    // Model Weight Tensors
    // ================================
    // Global model tensors (replicated across all devices)
    std::shared_ptr<Tensor> w_in_embd;   // Input embedding table [dvoc, d]
    std::shared_ptr<Tensor> w_out_norm;  // Final layer normalization [d]
    std::shared_ptr<Tensor> w_out_embd;  // Output embedding/LM head [d, dvoc]
    std::shared_ptr<Tensor> sin_table;   // RoPE sine table [dctx, dh/2]
    std::shared_ptr<Tensor> cos_table;   // RoPE cosine table [dctx, dh/2]
    
    // Per-layer attention weights (partitioned by heads across devices)
    std::vector<std::shared_ptr<Tensor>> w_attn_norm;    // Layer normalization [nlayer][d]
    std::vector<std::shared_ptr<Tensor>> w_attn_q_norm;  // Q normalization [nlayer][dh] (Qwen3-specific)
    std::vector<std::shared_ptr<Tensor>> w_attn_k_norm;  // K normalization [nlayer][dh] (Qwen3-specific)
    std::vector<std::shared_ptr<Tensor>> w_attn_q;       // Q projection [nlayer][d, d/ndev]
    std::vector<std::shared_ptr<Tensor>> w_attn_k;       // K projection [nlayer][d, (nkvh*dh)/ndev]
    std::vector<std::shared_ptr<Tensor>> w_attn_v;       // V projection [nlayer][d, (nkvh*dh)/ndev]
    std::vector<std::shared_ptr<Tensor>> w_attn_o;       // O projection [nlayer][d/ndev, d]
    
    // Per-layer MLP/MoE weights
    std::vector<std::shared_ptr<Tensor>> w_mlp_norm;     // MLP layer normalization [nlayer][d]
    
    // Regular MLP weights (for non-MoE layers)
    std::vector<std::shared_ptr<Tensor>> w_mlp_gate;     // MLP gate projection [nlayer][d, di/ndev]
    std::vector<std::shared_ptr<Tensor>> w_mlp_up;       // MLP up projection [nlayer][d, di/ndev]
    std::vector<std::shared_ptr<Tensor>> w_mlp_down;     // MLP down projection [nlayer][di/ndev, d]
    
    // MoE weights (for MoE layers)
    std::vector<std::shared_ptr<Tensor>> w_moe_gate;     // Router/gating weights [nlayer][d, num_experts]
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_gate;  // Expert gate proj [nlayer][num_experts_per_device][d, moe_di/ndev]
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_up;    // Expert up proj [nlayer][num_experts_per_device][d, moe_di/ndev]
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_down;  // Expert down proj [nlayer][num_experts_per_device][moe_di/ndev, d]
    
    // ================================
    // Layer Type Information
    // ================================
    std::vector<bool> is_moe_layer;      // Flag indicating which layers use MoE [nlayer]
    
    // ================================
    // MoE Runtime Statistics
    // ================================
    std::vector<std::vector<uint32_t>> expert_usage_count;  // Expert usage statistics [nlayer][num_experts]
    
    // ================================
    // Intermediate Computation Buffers
    // ================================
    std::shared_ptr<Tensor> hidden_states;      // Main hidden state buffer [max_batch_size * max_seq_len, d]
    std::shared_ptr<Tensor> residual_states;    // Residual connection buffer [max_batch_size * max_seq_len, d]
    std::shared_ptr<Tensor> attn_output;        // Attention output buffer [max_batch_size * max_seq_len, d]
    std::shared_ptr<Tensor> mlp_output;         // MLP/MoE output buffer [max_batch_size * max_seq_len, d]
    
    // MoE-specific computation buffers
    std::shared_ptr<Tensor> router_logits;      // Router logits [max_batch_size * max_seq_len, num_experts]
    std::shared_ptr<Tensor> routing_weights;    // Routing weights [max_batch_size * max_seq_len, num_experts_per_tok]
    std::shared_ptr<Tensor> selected_experts;   // Selected expert indices [max_batch_size * max_seq_len, num_experts_per_tok]
    std::shared_ptr<Tensor> expert_outputs;     // Combined expert outputs [max_batch_size * max_seq_len, d]
    std::shared_ptr<Tensor> expert_intermediate; // Expert intermediate buffer [max_tokens_per_expert, moe_intermediate_size/ndev]
};

/*
 * Qwen3-MoE模型主结构
 * 
 * 管理分布式Qwen3-MoE模型推理的完整状态：
 * - 元数据和配置参数
 * - 设备资源和权重分区
 * - 工作线程和同步原语
 * - 请求处理协调
 */
struct Qwen3MoeModel {
    // ================================
    // Model Configuration
    // ================================
    Qwen3MoeMeta meta;                   // Model architecture and MoE configuration
    
    // ================================
    // Distributed Resources
    // ================================
    size_t ndev;                         // Number of devices for distributed inference
    std::vector<DeviceQwen3MoeResource> dev_resources; // Per-device resources and weights
    std::vector<Qwen3MoeInferState> states;           // Per-device synchronization state
    std::vector<std::thread> threads;                 // Worker threads for async processing
    
    // ================================
    // Shared Request Processing
    // ================================
    Qwen3MoeInferRequest req;            // Shared request data across all devices
    
    // ================================
    // Constructor
    // ================================
    /*
     * Initialize distributed Qwen3-MoE model with tensor parallelism
     * 
     * Parameters:
     * - meta: Model architecture including MoE configuration
     * - weights: Model weight tensors including expert weights
     * - device: InfiniCore device type
     * - device_ids: Physical device IDs for distributed inference
     * 
     * Initialization Process:
     * 1. Create device resources and partition weights across devices
     * 2. Distribute experts across devices for load balancing
     * 3. Initialize InfiniCCL communication for multi-device coordination
     * 4. Launch worker threads and wait for device initialization
     * 5. Set up synchronization primitives for inference coordination
     */
    Qwen3MoeModel(const Qwen3MoeMeta *, const Qwen3MoeWeights *, infiniDevice_t device, std::vector<int> device_ids);
};

/*
 * Qwen3-MoE KV 缓存结构
 * 
 * 管理用于高效自回归生成的键值缓存存储，与基础Qwen3相同。
 * MoE层共享相同的注意力机制，因此KV缓存结构保持不变。
 * 
 * 缓存组织：
 * - 键（k）和值（v）的单独存储
 * - 为分布式推理在设备间分区 [ndev]
 * - 每个 transformer 层的单独缓存 [nlayer]
 * - 每个缓存张量的形状 [max_seq_len, nkvh/ndev, dh]
 */
struct Qwen3MoeKVCache {
    // ================================
    // Cache Storage Tensors
    // ================================
    std::vector<std::vector<std::shared_ptr<Tensor>>> k;  // Key cache [ndev][nlayer]
    std::vector<std::vector<std::shared_ptr<Tensor>>> v;  // Value cache [ndev][nlayer]
    /*
     * Cache Tensor Dimensions per Device per Layer:
     * Shape: [max_seq_len, nkvh/ndev, dh]
     * - max_seq_len: Maximum context length (dctx)
     * - nkvh/ndev: Key-value heads allocated to this device
     * - dh: Head dimension
     * 
     * Storage Layout in Memory:
     * - Contiguous storage allows efficient slicing for different sequence lengths
     * - Device partitioning enables parallel KV cache updates
     * - Layer separation allows independent cache management per transformer layer
     */
};

// Forward declarations for worker functions
void qwen3_moe_device_worker(
    DeviceQwen3MoeResource &resource,
    Qwen3MoeInferState &state,
    const Qwen3MoeMeta &meta,
    Qwen3MoeInferRequest &req,
    size_t device_id,
    size_t ndev);

#endif