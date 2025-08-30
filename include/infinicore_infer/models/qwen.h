#ifndef MODEL_QWEN_H
#define MODEL_QWEN_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct QwenModel;
struct KVCache; // 假设 KVCache 是一个通用的结构体

typedef struct
{
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
} QwenMeta;

typedef struct
{
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    // 0 if linear weights are passed as W, any other value if passed as W^T (default format in pytorch)
    int transpose_linear_weights;
    // [dvoc, d]
    const void *input_embd;
    // [d]
    const void *output_norm;
    // [dvoc, d]
    const void *output_embd;
    // nlayer * [d]
    const void *const *attn_norm;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh, d]
    const void *const *attn_qkv;
    // nlayer * [ndev, (nh + 2 * nkvh) / ndev * dh]
    const void *const *attn_qkv_b;
    // nlayer * [dh]
    const void *const *attn_q_norm;
    // nlayer * [dh]
    const void *const *attn_k_norm;
    // nlayer * [ndev, d, nkvh / ndev * dh]
    const void *const *attn_o;
    // nlayer * [d]
    const void *const *ffn_norm;
    // nlayer * [ndev, 2 * di / ndev, d]
    const void *const *ffn_gate_up;
    // nlayer * [ndev, d, di / ndev]
    const void *const *ffn_down;
} QwenWeights;


//////////////////// APIs ///////////////////////
/// @brief 创建模型
__C __export struct QwenModel*
createQwenModel(const QwenMeta*, const QwenWeights*, infiniDevice_t device, int ndev, const int* dev_ids);

/// @brief 销毁模型
__C __export void
destroyQwenModel(struct QwenModel*);

/// @brief 创建 KV Cache
// --- 修改：函数重命名 ---
__C __export struct KVCache*
createQwenKVCache(const struct QwenModel*);

/// @brief 复制 KV Cache
// --- 修改：函数重命名 ---
__C __export struct KVCache*
duplicateQwenKVCache(const struct QwenModel*, const struct KVCache*, uint32_t seq_len);

/// @brief 销毁 KV Cache
// --- 修改：函数重命名 ---
__C __export void
dropQwenKVCache(const struct QwenModel*, struct KVCache*);

/// @brief 批次推理一轮
// --- 修改：函数重命名 ---
__C __export void
inferQwenBatch(struct QwenModel*,
               const uint32_t* tokens, uint32_t ntok,
               const uint32_t* req_lens, uint32_t nreq, const uint32_t* req_pos,
               struct KVCache** kv_caches,
               const float* temperature, const uint32_t* topk, const float* topp,
               uint32_t* output);

#ifdef __cplusplus
}
#endif

#endif // MODEL_QWEN_H