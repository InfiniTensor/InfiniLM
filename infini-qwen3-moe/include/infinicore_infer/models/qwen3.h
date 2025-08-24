#ifndef MODEL_QWEN3_H
#define MODEL_QWEN3_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>
#include <cstddef>

struct Qwen3Model;

typedef struct
{
    /* ---- 数据类型 ---- */
    infiniDtype_t dt_logits;        // 权重/激活的浮点格式 → torch_dtype = bfloat16

    /* ---- 网络规模 ---- */
    size_t nlayer;                  // Transformer 层数 → num_hidden_layers = 28
    size_t d;                       // 隐藏维度 hidden_size = 2048
    size_t nh;                      // 查询头数 num_attention_heads = 16
    size_t nkvh;                    // 键/值头数 num_key_value_heads = 8
    size_t dh;                      // 单头维度 head_dim = 128
    size_t di;                      // mlp 中间维度 intermediate_size = 6144
    size_t dctx;                    // 最大上下文长度 max_position_embeddings = 40960
    size_t dvoc;                    // 词表大小 vocab_size = 151936

    /* ---- 归一化与 RoPE ---- */
    float epsilon;                  // RMSNorm ε rms_norm_eps = 1e-6
    float theta;                    // RoPE 基值 rope_theta = 1e6

    /* ---- 额外缺少的字段（新增） ---- */
    uint32_t bos_token;             // 起始符 bos_token_id = 151643
    uint32_t end_token;             // 结束符 eos_token_id = 151645
    float    attn_dropout;          // attention_dropout = 0.0 （推理时可忽略）
    bool     tie_embd;              // 输出头与嵌入共享 tie_word_embeddings = true
} Qwen3Meta;

typedef struct {
    /* ---- 元信息 ---- */
    size_t nlayer;
    infiniDtype_t dt_norm;/* 输入输出数据类型 */
    infiniDtype_t dt_mat;/* 矩阵数据类型 */
    int  transpose_linear_weights; /* 0: [in, out]  非0: [out, in] */

    /* ---- 全局共享 ---- */
    const void *input_embd;  /* [dvoc, d] 或 [d, dvoc] */
    const void *output_embd; /* 若 tie_word_embeddings==true，可与 input_embd 同址 */
    const void *output_norm; /* [d] */

    /* ---- 逐层权重（数组长度 = nlayer） ---- */
    const void **attn_norm;
    const void **attn_q_norm;/* [dh] */
    const void **attn_k_norm;/* [dh] */

    /* QKV 投影 —— 分开存放 */
    const void **attn_q_proj; // [d, d]
    const void **attn_k_proj; // [d, nkvh * dh]  (nkvh=8, dh=128 → 1024)
    const void **attn_v_proj; // [d, nkvh * dh]
    const void **attn_o_proj; // [d, d]
    const void **mlp_norm;    /* [nlayer] 每层: [d] */
    const void **mlp_gate_proj; /* [nlayer] 每层: [d, 2*di] 或转置 */
    const void **mlp_up_proj;   // [d, di]
    const void **mlp_down_proj;    /* [nlayer] 每层: [di, d] 或转置 */
} Qwen3Weights;

//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3Model *
createQwen3Model(const Qwen3Meta *,
                 const Qwen3Weights *,
                 infiniDevice_t device,
                 int ndev,
                 const int *dev_ids);

/// @brief 销毁模型
__C __export void
destroyQwen3Model(struct Qwen3Model *);

/// @brief 创建 KV Cache
__C __export struct Qwen3KVCache *
createQwen3KVCache(const struct Qwen3Model *);

/// @brief 复制 KV Cache
__C __export struct Qwen3KVCache *
duplicateQwen3KVCache(const struct Qwen3Model *,
                      const struct Qwen3KVCache *, uint32_t seq_len);

/// @brief 销毁 KV Cache
__C __export void
dropQwen3KVCache(const struct Qwen3Model *,
                 struct Qwen3KVCache *);

/// @brief 批次推理一轮
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
inferQwen3Batch(struct Qwen3Model *,
           const uint32_t *tokens, uint32_t ntok,
           const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
           struct Qwen3KVCache **kv_caches,
           const float *temperature, const uint32_t *topk, const float *topp,
           uint32_t *output);

/// @brief 启用或禁用调试模式（保存中间张量到文件）
__C __export void
setQwen3DebugMode(int enabled);

#endif
