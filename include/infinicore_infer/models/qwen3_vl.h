#ifndef MODEL_QWEN3_VL_H
#define MODEL_QWEN3_VL_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

#include "../weights_loader.h"

struct Qwen3VLModel;

typedef struct
{
    infiniDtype_t dt_logits;
    infiniDtype_t dt_linear_w;
    infiniDtype_t dt_norm_w;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
    char has_qkv_bias;
    char use_qk_norm;
    // Vision encoder parameters
    size_t vision_hidden_size;
    size_t vision_layers;
    size_t vision_heads;
    size_t patch_size;
    size_t img_size;
    // Token ids
    uint32_t image_token_id;
    uint32_t video_token_id;
} Qwen3VLMeta;

//////////////////// APIs ///////////////////////
__C __export struct ModelWeights *
createQwen3VLWeights(const Qwen3VLMeta *,
                     infiniDevice_t device,
                     int ndev,
                     const int *dev_ids);

/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3VLModel *
createQwen3VLModel(const Qwen3VLMeta *,
                   const ModelWeights *);

/// @brief 销毁模型
__C __export void
destroyQwen3VLModel(struct Qwen3VLModel *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param pos_ids ViT位置编码，格式[patches, 2] (h,w)
/// @param pos_ids_len pos_ids数组长度，应为patches*2
/// @param llm_pos_ids LLM 3D mRoPE位置编码，格式[patches+text_len, 3] (t,h,w)
/// @param llm_pos_ids_len llm_pos_ids数组长度，应为(patches+text_len)*3
/// @param rope_section 3D mRoPE区段配置，格式[3] (t_max,h_max,w_max)
/// @param rope_section_len rope_section数组长度，应为3
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferBatchQwen3VL(struct Qwen3VLModel *,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  const uint32_t *pos_ids, uint32_t pos_ids_len,
                  const uint32_t *llm_pos_ids, uint32_t llm_pos_ids_len,
                  const uint32_t *rope_section, uint32_t rope_section_len,
                  const float *pixel_values,
                  struct KVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param pos_ids ViT位置编码，格式[patches, 2] (h,w)
/// @param pos_ids_len pos_ids数组长度，应为patches*2
/// @param llm_pos_ids LLM 3D mRoPE位置编码，格式[patches+text_len, 3] (t,h,w)
/// @param llm_pos_ids_len llm_pos_ids数组长度，应为(patches+text_len)*3
/// @param rope_section 3D mRoPE区段配置，格式[3] (t_max,h_max,w_max)
/// @param rope_section_len rope_section数组长度，应为3
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
forwardBatchQwen3VL(struct Qwen3VLModel *,
                    const uint32_t *tokens, uint32_t ntok,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    const uint32_t *pos_ids, uint32_t pos_ids_len,
                    const uint32_t *llm_pos_ids, uint32_t llm_pos_ids_len,
                    const uint32_t *rope_section, uint32_t rope_section_len,
                    const float *pixel_values,
                    struct KVCache **kv_caches,
                    void *logits);

#endif
