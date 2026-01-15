#ifndef MODEL_JIUGE_H
#define MODEL_JIUGE_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

struct JiugeModel;

typedef struct
{
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
} JiugeMeta;

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
    const void *const *attn_norm;  // 指针数组，每层一个RMSNorm权重
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
} JiugeWeights;

//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct JiugeModel *
createJiugeModel(const JiugeMeta *,
                 const JiugeWeights *,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

/// @brief 销毁模型
__C __export void
destroyJiugeModel(struct JiugeModel *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferBatchJiuge(struct JiugeModel *,
                const uint32_t *tokens, uint32_t ntok,
                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                struct KVCache **kv_caches,
                const float *temperature, const uint32_t *topk, const float *topp,
                uint32_t *output);

/// @brief 批次推理一轮，并采样出新的 token（RoPE 位置与 KV 写入位置可解耦，用于 KV 压缩）
/// @param req_pos 位置 id 基址（用于 RoPE/pos_ids 计算）
/// @param kv_pos KVCache 写入/读取基址（用于 past_len/total_len 计算）
__C __export void
inferBatchJiugeEx(struct JiugeModel *,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq,
                  const uint32_t *req_pos,
                  const uint32_t *kv_pos,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output);

/// @brief 批次推理一轮，并采样出新的 token，同时输出 logits
/// @param logits 输出 logits 数组
__C __export void
inferBatchJiugeWithLogits(struct JiugeModel *,
                         const uint32_t *tokens, uint32_t ntok,
                         const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                         struct KVCache **kv_caches,
                         const float *temperature, const uint32_t *topk, const float *topp,
                         uint32_t *output, void *logits);

/// @brief 批次推理一轮（RoPE 位置与 KV 写入位置可解耦），同时输出 logits
/// @param req_pos 位置 id 基址（用于 RoPE/pos_ids 计算）
/// @param kv_pos KVCache 写入/读取基址（用于 past_len/total_len 计算）
/// @param logits 输出 logits 数组
__C __export void
inferBatchJiugeExWithLogits(struct JiugeModel *,
                            const uint32_t *tokens, uint32_t ntok,
                            const uint32_t *req_lens, uint32_t nreq,
                            const uint32_t *req_pos,
                            const uint32_t *kv_pos,
                            struct KVCache **kv_caches,
                            const float *temperature, const uint32_t *topk, const float *topp,
                            uint32_t *output, void *logits);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
forwardBatchJiuge(struct JiugeModel *,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct KVCache **kv_caches,
                  void *logits);

/// @brief 批次推理一轮，输出 logits（RoPE 位置与 KV 写入位置可解耦，用于 KV 压缩）
__C __export void
forwardBatchJiugeEx(struct JiugeModel *,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq,
                    const uint32_t *req_pos,
                    const uint32_t *kv_pos,
                    struct KVCache **kv_caches,
                    void *logits);

/// @brief 批次推理一轮，支持对指定 token 位置的输入 embedding 做覆盖（用于多模态 image embedding 注入）
/// @note override_pos 需要按升序排列，且每个位置最多出现一次
/// @param n_override 覆盖位置数量
/// @param override_pos 覆盖位置（基于拼接后的 tokens 序列下标，范围 [0, ntok)）
/// @param override_embeds 覆盖 embedding，shape [n_override, d]，dtype = meta.dt_logits
__C __export void
inferBatchJiugeWithOverrides(struct JiugeModel *,
                             const uint32_t *tokens, uint32_t ntok,
                             const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                             struct KVCache **kv_caches,
                             uint32_t n_override,
                             const uint32_t *override_pos,
                             const void *override_embeds,
                             const float *temperature, const uint32_t *topk, const float *topp,
                             uint32_t *output);

/// @brief 批次推理一轮（RoPE 位置与 KV 写入位置可解耦），支持 embedding 覆盖
__C __export void
inferBatchJiugeWithOverridesEx(struct JiugeModel *,
                               const uint32_t *tokens, uint32_t ntok,
                               const uint32_t *req_lens, uint32_t nreq,
                               const uint32_t *req_pos,
                               const uint32_t *kv_pos,
                               struct KVCache **kv_caches,
                               uint32_t n_override,
                               const uint32_t *override_pos,
                               const void *override_embeds,
                               const float *temperature, const uint32_t *topk, const float *topp,
                               uint32_t *output);

/// @brief 批次推理一轮，支持 embedding 覆盖，同时输出 logits
__C __export void
inferBatchJiugeWithOverridesWithLogits(struct JiugeModel *,
                                       const uint32_t *tokens, uint32_t ntok,
                                       const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                                       struct KVCache **kv_caches,
                                       uint32_t n_override,
                                       const uint32_t *override_pos,
                                       const void *override_embeds,
                                       const float *temperature, const uint32_t *topk, const float *topp,
                                       uint32_t *output, void *logits);

// /// @brief 批次推理一轮（RoPE 位置与 KV 写入位置可解耦），支持 embedding 覆盖，同时输出 logits
// __C __export void
// inferBatchJiugeWithOverridesExWithLogits(struct JiugeModel *,
//                                           const uint32_t *tokens, uint32_t ntok,
//                                           const uint32_t *req_lens, uint32_t nreq,
//                                           const uint32_t *req_pos,
//                                           const uint32_t *kv_pos,
//                                           struct KVCache **kv_caches,
//                                           uint32_t n_override,
//                                           const uint32_t *override_pos,
//                                           const void *override_embeds,
//                                           const float *temperature, const uint32_t *topk, const float *topp,
//                                           uint32_t *output, void *logits);

/// @brief 批次推理一轮，输出 logits，支持对指定 token 位置的输入 embedding 做覆盖
__C __export void
forwardBatchJiugeWithOverrides(struct JiugeModel *,
                               const uint32_t *tokens, uint32_t ntok,
                               const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                               struct KVCache **kv_caches,
                               uint32_t n_override,
                               const uint32_t *override_pos,
                               const void *override_embeds,
                               void *logits);

/// @brief 批次推理一轮，输出 logits（RoPE 位置与 KV 写入位置可解耦），支持 embedding 覆盖
__C __export void
forwardBatchJiugeWithOverridesEx(struct JiugeModel *,
                                 const uint32_t *tokens, uint32_t ntok,
                                 const uint32_t *req_lens, uint32_t nreq,
                                 const uint32_t *req_pos,
                                 const uint32_t *kv_pos,
                                 struct KVCache **kv_caches,
                                 uint32_t n_override,
                                 const uint32_t *override_pos,
                                 const void *override_embeds,
                                 void *logits);

#endif
