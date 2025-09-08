#ifndef MODEL_QWEN_HYBRID_H
#define MODEL_QWEN_HYBRID_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include "../cache.h"
#include "../weights_loader.h"

#include <stdint.h>

struct QwenHybridModel;

typedef struct
{
    infiniDtype_t dt_logits;
    infiniDtype_t dt_linear_w;
    infiniDtype_t dt_norm_w;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
    char has_qkv_bias;
} QwenHybridMeta;

//////////////////// APIs ///////////////////////
__C __export struct ModelWeights *
createQwenHybridWeights(const QwenHybridMeta *,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids);
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct QwenHybridModel *
createQwenHybridModel(const QwenHybridMeta *,
                      const ModelWeights *);

/// @brief 销毁模型
__C __export void
destroyQwenHybridModel(struct QwenHybridModel *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param mamba_caches 每个请求的 Mamba Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferBatchQwenHybrid(struct QwenHybridModel *,
                     const uint32_t *tokens, uint32_t ntok,
                     const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                     struct KVCache **kv_caches,
                     struct MambaCache **mamba_caches,
                     const float *temperature, const uint32_t *topk, const float *topp,
                     uint32_t *output);

#endif
