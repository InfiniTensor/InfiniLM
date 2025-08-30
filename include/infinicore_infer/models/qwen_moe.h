#ifndef MODEL_QWEN_MOE_H
#define MODEL_QWEN_MOE_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration for the new MoE model handle
struct QwenMoeModel;
// KVCache struct can be reused if its definition is generic enough,
// otherwise it should also be specialized. Assuming it's generic for now.
struct KVCache;


// Renamed and specialized Meta struct for MoE
typedef struct
{
    // --- Standard Fields (same as dense model) ---
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;

    // --- New MoE-Specific Fields ---
    size_t num_experts;         // Total number of experts per layer
    size_t num_experts_per_tok; // Number of active experts per token
    size_t moe_intermediate_size; // Intermediate size of a single expert's FFN
    int norm_topk_prob;         // Flag (0 or 1) for routing logic
    
} QwenMoeMeta;

// Renamed and redesigned Weights struct for MoE
typedef struct
{
    // --- Standard Fields (same as dense model) ---
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    int transpose_linear_weights;
    const void *input_embd;       // [dvoc, d]
    const void *output_norm;      // [d]
    const void *output_embd;      // [dvoc, d]

    // --- Attention Block (same as dense model) ---
    const void *const *attn_norm;     // nlayer * [d]
    const void *const *attn_qkv;      // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh, d]
    const void *const *attn_qkv_b;    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh]
    const void *const *attn_q_norm;   // nlayer * [dh]
    const void *const *attn_k_norm;   // nlayer * [dh]
    const void *const *attn_o;        // nlayer * [ndev, d, nkvh / ndev * dh]

    // --- MoE Block (replaces dense FFN) ---
    const void *const *ffn_norm;      // Still needed: nlayer * [d] (post_attention_layernorm)
    
    // Pointers for the Gating Network in each layer
    const void *const *moe_gate;      // nlayer * [num_experts, d]

    // Pointers for the Experts. These point to flattened arrays of pointers.
    // The total length of each array is (nlayer * num_experts).
    // Access in C++ via: array[layer_idx * num_experts + expert_idx]
    const void *const *moe_experts_gate_up; // Flat array of pointers to each expert's gate_up/swiglu weights
    const void *const *moe_experts_down;    // Flat array of pointers to each expert's down_proj weights

} QwenMoeWeights;


//////////////////// New MoE APIs ///////////////////////
/// @brief 创建 MoE 模型
__C __export struct QwenMoeModel *
createQwenMoeModel(const QwenMoeMeta *,
                   const QwenMoeWeights *,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids);

/// @brief 销毁 MoE 模型
__C __export void
destroyQwenMoeModel(struct QwenMoeModel *);

/// @brief 为 MoE 模型创建 KV Cache
__C __export struct KVCache *
createQwenMoeKVCache(const struct QwenMoeModel *);

/// @brief 为 MoE 模型复制 KV Cache
__C __export struct KVCache *
duplicateQwenMoeKVCache(const struct QwenMoeModel *,
                        const struct KVCache *, uint32_t seq_len);

/// @brief 为 MoE 模型销毁 KV Cache
__C __export void
dropQwenMoeKVCache(const struct QwenMoeModel *,
                   struct KVCache *);

/// @brief MoE 模型批次推理一轮，并采样出新的 token
__C __export void
inferQwenMoeBatch(struct QwenMoeModel *,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output);

#ifdef __cplusplus
}
#endif

#endif // MODEL_QWEN_MOE_H