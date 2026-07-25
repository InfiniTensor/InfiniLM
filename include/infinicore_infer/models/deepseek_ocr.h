#ifndef MODEL_DEEPSEEK_OCR_H
#define MODEL_DEEPSEEK_OCR_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

struct DeepSeekOCRModel;

typedef struct
{
    infiniDtype_t dt_logits;
    infiniDtype_t dt_norm;
    // Layer counts
    size_t n_dense_layer;  // 第0层是dense
    size_t n_sparse_layer; // 第1-11层是MoE
    // Model dimensions
    size_t d;    // hidden_size: 1280
    size_t nh;   // num_attention_heads: 1280
    size_t nkvh; // num_key_value_heads: 1280
    size_t dh;   // head_dim: d/nh = 1
    // Dense MLP dimensions
    size_t di_dense; // intermediate_size for dense layer: 6848
    // MoE dimensions
    size_t di_moe;      // moe_intermediate_size: 896
    size_t di_shared;   // shared_expert_intermediate_size: 1792
    size_t nexperts;    // n_routed_experts: 64
    size_t kexperts;    // num_experts_per_tok: 6
    float routed_scale; // routed_scaling_factor: 1.0
    // Context and vocab
    size_t dctx; // max_position_embeddings
    size_t dvoc; // vocab_size: 129280
    // Normalization
    float epsilon;      // rms_norm_eps: 1e-6
    float theta;        // rope_theta: 10000.0
    uint32_t end_token; // eos_token_id
} DeepSeekOCRMeta;

typedef struct
{
    size_t n_dense_layer;
    size_t n_sparse_layer;
    infiniDtype_t dt_norm, dt_mat;
    // 0 if linear weights are passed as W, any other value if passed as W^T
    int transpose_linear_weights;

    // Embeddings
    const void *input_embd;  // [dvoc, d]
    const void *output_norm; // [d]
    const void *output_embd; // [dvoc, d]

    // Attention layers (all layers: n_dense_layer + n_sparse_layer)
    const void *const *attn_norm; // nlayer * [d]
    const void *const *attn_q;    // nlayer * [d, d] or sharded
    const void *const *attn_k;    // nlayer * [d, d] or sharded
    const void *const *attn_v;    // nlayer * [d, d] or sharded
    const void *const *attn_o;    // nlayer * [d, d] or sharded

    // FFN layers
    const void *const *ffn_norm; // nlayer * [d]

    // Dense MLP (layer 0)
    const void *dense_gate; // [di_dense, d]
    const void *dense_up;   // [di_dense, d]
    const void *dense_down; // [d, di_dense]

    // MoE layers (layer 1-11)
    const void *const *moe_gate_weight; // n_sparse_layer * [nexperts, d]
    const void *const *moe_gate_bias;   // n_sparse_layer * [nexperts]

    // Shared experts
    const void *const *moe_shared_gate; // n_sparse_layer * [di_shared, d]
    const void *const *moe_shared_up;   // n_sparse_layer * [di_shared, d]
    const void *const *moe_shared_down; // n_sparse_layer * [d, di_shared]

    // Routed experts
    const void *const *const *moe_experts_gate; // n_sparse_layer * nexperts * [di_moe, d]
    const void *const *const *moe_experts_up;   // n_sparse_layer * nexperts * [di_moe, d]
    const void *const *const *moe_experts_down; // n_sparse_layer * nexperts * [d, di_moe]

    // Vision Encoder weights
    // SAM ViT-B
    const void *sam_patch_embed;
    const void *sam_patch_embed_bias;
    const void *const *sam_block_norm1;     // 12 layers
    const void *const *sam_block_attn_qkv;  // 12 layers
    const void *const *sam_block_attn_proj; // 12 layers
    const void *const *sam_block_norm2;     // 12 layers
    const void *const *sam_block_mlp_fc1;   // 12 layers
    const void *const *sam_block_mlp_fc2;   // 12 layers
    const void *sam_neck_conv1;
    const void *sam_neck_ln1;
    const void *sam_neck_conv2;
    const void *sam_neck_ln2;

    // CLIP-L
    const void *clip_patch_embed;
    const void *clip_patch_embed_bias;
    const void *clip_position_embed;
    const void *clip_pre_layernorm;
    const void *const *clip_block_ln1;       // 24 layers
    const void *const *clip_block_attn_qkv;  // 24 layers
    const void *const *clip_block_attn_proj; // 24 layers
    const void *const *clip_block_ln2;       // 24 layers
    const void *const *clip_block_mlp_fc1;   // 24 layers
    const void *const *clip_block_mlp_fc2;   // 24 layers

    // Projector
    const void *projector;      // [2048, 1280] Linear projection
    const void *image_newline;  // [1280] Image row separator
    const void *view_seperator; // [1280] View separator
} DeepSeekOCRWeights;

//////////////////// APIs ///////////////////////

/// @brief 创建DeepSeek-OCR模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct DeepSeekOCRModel *
createDeepSeekOCRModel(const DeepSeekOCRMeta *,
                       const DeepSeekOCRWeights *,
                       infiniDevice_t device,
                       int ndev,
                       const int *dev_ids);

/// @brief 销毁模型
__C __export void
destroyDeepSeekOCRModel(struct DeepSeekOCRModel *);

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
inferBatchDeepSeekOCR(struct DeepSeekOCRModel *,
                      const uint32_t *tokens, uint32_t ntok,
                      const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature, const uint32_t *topk, const float *topp,
                      uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 logits，shape: [ntok, dvoc]
__C __export void
forwardBatchDeepSeekOCR(struct DeepSeekOCRModel *,
                        const uint32_t *tokens, uint32_t ntok,
                        const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                        struct KVCache **kv_caches,
                        void *logits);

/// @brief 使用预计算的embeddings进行推理(用于多模态输入)
/// @param inputs_embeds 输入embeddings，shape: [ntok, d]
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度
/// @param topk 采样 topk
/// @param topp 采样 topp
/// @param output 输出 token 数组
__C __export void
inferBatchDeepSeekOCRWithEmbeds(struct DeepSeekOCRModel *,
                                const void *inputs_embeds, uint32_t ntok,
                                const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                                struct KVCache **kv_caches,
                                const float *temperature, const uint32_t *topk, const float *topp,
                                uint32_t *output);

#endif
