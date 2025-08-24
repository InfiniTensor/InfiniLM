#ifndef MODEL_QWEN3_MOE_H
#define MODEL_QWEN3_MOE_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>
#include <cstddef>

struct Qwen3MoeModel;

typedef struct
{
    /* ---- 数据类型 ---- */
    infiniDtype_t dt_logits;        // 权重/激活的浮点格式

    /* ---- 网络规模 ---- */
    size_t nlayer;                  // Transformer 层数
    size_t d;                       // 隐藏维度 hidden_size
    size_t nh;                      // 查询头数 num_attention_heads
    size_t nkvh;                    // 键/值头数 num_key_value_heads
    size_t dh;                      // 单头维度 head_dim
    size_t di;                      // mlp 中间维度 intermediate_size
    size_t dctx;                    // 最大上下文长度 max_position_embeddings
    size_t dvoc;                    // 词表大小 vocab_size

    /* ---- 归一化与 RoPE ---- */
    float epsilon;                  // RMSNorm ε rms_norm_eps
    float theta;                    // RoPE 基值 rope_theta

    /* ---- 基础字段 ---- */
    uint32_t bos_token;             // 起始符 bos_token_id
    uint32_t end_token;             // 结束符 eos_token_id
    float    attn_dropout;          // attention_dropout
    bool     tie_embd;              // 输出头与嵌入共享 tie_word_embeddings

    /* ---- MoE 特有参数 ---- */
    size_t num_experts;             // 每个MoE层的专家数量
    size_t num_experts_per_tok;     // 每个token选择的专家数量 (top-k)
    size_t moe_intermediate_size;   // MoE专家的中间维度
    size_t decoder_sparse_step;     // MoE层的间隔步长
    size_t num_mlp_only_layers;     // 纯MLP层的数量
    size_t *mlp_only_layers;        // 纯MLP层的索引数组
    bool norm_topk_prob;            // 是否对top-k概率进行归一化
    float router_aux_loss_coef;     // 路由辅助损失系数
} Qwen3MoeMeta;

typedef struct {
    /* ---- 元信息 ---- */
    size_t nlayer;
    infiniDtype_t dt_norm;          // 归一化权重数据类型
    infiniDtype_t dt_mat;           // 矩阵权重数据类型
    int transpose_linear_weights;   // 0: [in, out]  非0: [out, in]

    /* ---- 全局共享权重 ---- */
    const void *input_embd;         // [dvoc, d] 或 [d, dvoc]
    const void *output_embd;        // 若 tie_word_embeddings==true，可与 input_embd 同址
    const void *output_norm;        // [d]

    /* ---- 逐层权重（数组长度 = nlayer） ---- */
    const void **attn_norm;         // [nlayer] 每层: [d]
    const void **attn_q_norm;       // [nlayer] 每层: [dh] - Qwen3特有的Q/K归一化
    const void **attn_k_norm;       // [nlayer] 每层: [dh] - Qwen3特有的Q/K归一化

    /* QKV 投影 - 分开存放 */
    const void **attn_q_proj;       // [nlayer] 每层: [d, d]
    const void **attn_k_proj;       // [nlayer] 每层: [d, nkvh * dh]
    const void **attn_v_proj;       // [nlayer] 每层: [d, nkvh * dh]
    const void **attn_o_proj;       // [nlayer] 每层: [d, d]
    
    /* ---- MLP/MoE 权重 ---- */
    const void **mlp_norm;          // [nlayer] 每层: [d]
    
    /* MLP 权重 (for non-MoE layers) */
    const void **mlp_gate_proj;     // [nlayer] 每层: [d, di] - 非MoE层使用
    const void **mlp_up_proj;       // [nlayer] 每层: [d, di] - 非MoE层使用  
    const void **mlp_down_proj;     // [nlayer] 每层: [di, d] - 非MoE层使用

    /* MoE 权重 (for MoE layers) */
    const void **moe_gate;          // [nlayer] 每层: [d, num_experts] - 路由门控网络
    const void ***moe_experts_gate_proj;  // [nlayer][num_experts] 每专家: [d, moe_intermediate_size]
    const void ***moe_experts_up_proj;    // [nlayer][num_experts] 每专家: [d, moe_intermediate_size]  
    const void ***moe_experts_down_proj;  // [nlayer][num_experts] 每专家: [moe_intermediate_size, d]

    /* ---- MoE 元信息 ---- */
    size_t num_experts;             // 专家数量
    size_t num_experts_per_tok;     // 每token选择的专家数
    size_t moe_intermediate_size;   // MoE专家中间维度
    size_t decoder_sparse_step;     // MoE层间隔
    size_t num_mlp_only_layers;     // 纯MLP层数量
    size_t *mlp_only_layers;        // 纯MLP层索引
    bool norm_topk_prob;            // 是否归一化top-k概率
} Qwen3MoeWeights;

//////////////////// APIs ///////////////////////
/// @brief 创建Qwen3-MoE模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3MoeModel *
createQwen3MoeModel(const Qwen3MoeMeta *,
                   const Qwen3MoeWeights *,
                   infiniDevice_t device,
                   int ndev,
                   const int *dev_ids);

/// @brief 销毁Qwen3-MoE模型
__C __export void
destroyQwen3MoeModel(struct Qwen3MoeModel *);

/// @brief 创建Qwen3-MoE KV Cache
__C __export struct Qwen3MoeKVCache *
createQwen3MoeKVCache(const struct Qwen3MoeModel *);

/// @brief 复制Qwen3-MoE KV Cache
__C __export struct Qwen3MoeKVCache *
duplicateQwen3MoeKVCache(const struct Qwen3MoeModel *,
                        const struct Qwen3MoeKVCache *, uint32_t seq_len);

/// @brief 销毁Qwen3-MoE KV Cache
__C __export void
dropQwen3MoeKVCache(const struct Qwen3MoeModel *,
                   struct Qwen3MoeKVCache *);

/// @brief Qwen3-MoE批次推理一轮
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
inferQwen3MoeBatch(struct Qwen3MoeModel *,
                  const uint32_t *tokens, uint32_t ntok,
                  const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                  struct Qwen3MoeKVCache **kv_caches,
                  const float *temperature, const uint32_t *topk, const float *topp,
                  uint32_t *output);

/// @brief 启用或禁用Qwen3-MoE调试模式（保存中间张量到文件）
__C __export void
setQwen3MoeDebugMode(int enabled);

/// @brief 获取MoE路由统计信息（用于负载均衡分析）
/// @param model Qwen3-MoE模型实例
/// @param layer_idx 层索引
/// @param expert_counts 输出每个专家的使用次数，长度为 num_experts
__C __export void
getQwen3MoeRouterStats(const struct Qwen3MoeModel *model,
                       size_t layer_idx,
                       uint32_t *expert_counts);

#endif