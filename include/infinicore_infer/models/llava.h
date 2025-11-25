#ifndef MODEL_LLAVA_H
#define MODEL_LLAVA_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

struct LlavaModel;

// Vision Encoder Meta
typedef struct {
    size_t image_size;
    size_t patch_size;
    size_t num_patches;
    size_t vision_embed_dim;
    size_t vision_num_layers;
    size_t vision_num_heads;
    size_t vision_intermediate_size;
    float vision_epsilon;
} LlavaVisionMeta;

// Language Model Meta (reuses Jiuge structure)
typedef struct {
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
} LlavaLanguageMeta;

// MultiModal Projector Meta
typedef struct {
    size_t vision_embed_dim;
    size_t text_embed_dim;
    size_t projector_hidden_size;
} LlavaProjectorMeta;

typedef struct {
    LlavaVisionMeta vision_meta;
    LlavaLanguageMeta language_meta;
    LlavaProjectorMeta projector_meta;
} LlavaMeta;

typedef struct {
    // Vision Encoder Weights
    const void *vision_patch_embed_weight;  // [num_patches, vision_embed_dim]
    const void *vision_patch_embed_bias;    // [vision_embed_dim]
    const void *const *vision_encoder_weights;  // vision_layers * [various vision weights]

    // MultiModal Projector Weights
    const void *projector_weight;  // [text_embed_dim, vision_embed_dim]
    const void *projector_bias;    // [text_embed_dim]

    // Language Model Weights (reuses Jiuge structure)
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    int transpose_linear_weights;

    // Language model weights
    const void *input_embd;
    const void *output_norm;
    const void *output_embd;
    const void *const *attn_norm;
    const void *const *attn_qkv;
    const void *const *attn_qkv_b;
    const void *const *attn_q_norm;
    const void *const *attn_k_norm;
    const void *const *attn_o;
    const void *const *ffn_norm;
    const void *const *ffn_gate_up;
    const void *const *ffn_down;
} LlavaWeights;

struct LlavaKVCache;

//////////////////// APIs ///////////////////////
/// @brief 创建LLaVA模型
/// @param vision_meta 视觉编码器元信息
/// @param language_meta 语言模型元信息
/// @param projector_meta 多模态投影器元信息
/// @param weights 模型权重
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct LlavaModel *
createLlavaModel(const LlavaMeta *meta,
                 const LlavaWeights *weights,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

/// @brief 销毁LLaVA模型
/// @param model 模型实例
__C __export void
destroyLlavaModel(struct LlavaModel *model);

/// @brief 视觉编码前向推理
/// @param model 模型实例
/// @param image_tensor 输入图像张量
/// @param output 输出视觉特征
__C __export void
encodeVision(struct LlavaModel *model,
             const void *image_tensor,
             void *output);

/// @brief 批量视觉编码推理（用于Python接口）
/// @param model 模型实例
/// @param image_data 图像数据指针
/// @param output 输出缓冲区
__C __export void
inferBatchLlavaVison(struct LlavaModel *model,
                   const void *image_data,
                   void *output);

/// @brief 多模态投影前向推理
/// @param model 模型实例
/// @param vision_features 视觉特征
/// @param output 投影后的文本嵌入
__C __export void
projectMultiModal(struct LlavaModel *model,
                  const void *vision_features,
                  void *output);

/// @brief 语言模型批处理推理 (复用Jiuge逻辑)
/// @param model 模型实例
/// @param tokens 输入tokens
/// @param ntok tokens长度
/// @param req_lens 每个请求长度
/// @param nreq 请求数量
/// @param req_pos 每个请求当前的位置
/// @param kv_caches KV缓存
/// @param temperature 温度参数
/// @param topk Top-K采样参数
/// @param topp Top-P采样参数
/// @param output 输出token IDs
__C __export void
inferBatchLlavaLanguage(struct LlavaModel *model,
                       const uint32_t *tokens,
                       size_t ntok,
                       const size_t *req_lens,
                       size_t nreq,
                       const size_t *req_pos,
                       struct LlavaKVCache *kv_caches,
                       const float *temperature,
                       const float *topk,
                       const float *topp,
                       uint32_t *output);

#endif