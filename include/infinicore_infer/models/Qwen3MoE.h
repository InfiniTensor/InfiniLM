#ifndef QWEN3MOE
#define QWEN3MOE

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

struct Qwen3MoEWeights;

/// @brief 函数指针
typedef void (*load_global)(Qwen3MoEWeights *, void *cpu_ptr);
typedef void (*load_layer)(Qwen3MoEWeights *, void *cpu_ptr, size_t layer_id);
typedef void (*load_layer_linear)(Qwen3MoEWeights *, void *weight_ptr, size_t layer_id);
/// @brief 权重加载器
typedef struct {
    // Pre-Norm
    load_layer load_attn_norm;

    // Attention
    load_layer_linear load_attn_q_proj; 
    load_layer_linear load_attn_k_proj; 
    load_layer_linear load_attn_v_proj; 

    // QKNorm(RMSNorm)
    load_layer load_attn_q_norm;
    load_layer load_attn_k_norm;

    // output linear
    load_layer_linear load_attn_o_proj;

}Qwen3MoEWeightLoader;

struct Qwen3MoEAttention;

/// @brief 模型参数
typedef struct {
    //数据种类 BF16 / FP16
    infiniDtype_t dtype;

    // Linear args
    size_t hidden_size;
    size_t num_heads;
    size_t num_kv_head; // k_v head GQA广播倍数
    size_t head_dim;

    // RoPE args
    float rope_theta;
    size_t max_seq_len; 

    float rms_norm_eps; //防止除零
}Qwen3MoEAttentionMeta;

/// ==================== API ====================

/// @brief 创建注意力模块
__C __export struct Qwen3MoEAttention *
createQwen3MoEAttention(const Qwen3MoEAttentionMeta *,
                        const Qwen3MoEWeights *);
/// @brief 创建权重矩阵
__C Qwen3MoEWeights *
createQwen3MoEWeights(const Qwen3MoEAttentionMeta *meta,
                        infiniDevice_t device,
                        int ndev,             
                        const int *dev_ids);  
/// @brief 创建weight加载器
__C __export Qwen3MoEWeightLoader *
createQwen3MoEWeightLoader();
/// @brief 创建KVCache
__C __export struct Qwen3Cache *
createQwen3Cache(const Qwen3MoEAttentionMeta *meta,
                    size_t batch_size, size_t seq_len);
/// @brief 前向计算
__C __export void forwardQwen3MoEAttention(
    struct Qwen3MoEAttention* context,
    struct Qwen3Cache* kv_cache,
    const void* input_tensor,
    void* output_tensor,
    int batch_size,           // [新增]
    const int* seq_lens_ptr,  // [新增]
    const int* past_lens_ptr, // [新增]
    const int* pos_ids_ptr    // [新增]
);

/// @brief 销毁模型
__C __export void destroyQwen3MoEAttention(struct Qwen3MoEAttention* ctx);


#endif 