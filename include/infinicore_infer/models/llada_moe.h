#ifndef MODEL_LLADAMOE_H
#define MODEL_LLADAMOE_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stddef.h>
#include <stdint.h>

struct LLaDAMoEModel; 

typedef struct {
// 数据类型配置
    infiniDtype_t dt_logits;          // 对应 torch_dtype: "bfloat16"
    infiniDtype_t dt_norm;            // 通常使用FP32保证精度
    infiniDtype_t dt_quant_weight;    // 根据量化策略设置
    infiniDtype_t dt_quant_scale;     // 根据量化策略设置  
    infiniDtype_t dt_quant_zero;      // 根据量化策略设置
    infiniDtype_t dt_gate_weight;     // 对应 bfloat16
    infiniDtype_t dt_gate_bias;       // 对应 bfloat16

    // 模型维度参数
    size_t n_sparse_layer;    // 16 (所有层都是MoE层)
    size_t n_dense_layer;     // 0 (没有纯稠密层)
    size_t d;                 // 2048 (hidden_size)
    size_t nh;                // 16 (num_attention_heads)
    size_t nkvh;              // 16 (num_key_value_heads)
    size_t d_rope;            // 128 (根据partial_rotary_factor=1计算)
    size_t d_nope;            // 1920 (d - d_rope)
    size_t r_q;               // 可能需要从模型代码中获取
    size_t r_kv;              // 可能需要从模型代码中获取
    size_t d_qk;              // 128 (d / nh = 2048/16)
    size_t d_v;               // 128 (d / nh = 2048/16)

    float routed_scale;       // 1.0 (routed_scaling_factor)
    size_t nexperts;          // 64 (num_experts)
    size_t kexperts;          // 8 (num_experts_per_tok)
    size_t di;                // 2048 (hidden_size)
    size_t di_moe;           // 1024 (expert_intermediate_size)
    size_t dctx;             // 8192 (max_position_embeddings)
    size_t dvoc;             // 157184 (vocab_size)

    float epsilon;           // 1e-05 (rms_norm_eps)
    float rope_theta;        // 50000.0 (rope_theta)
    uint32_t end_token;      // 156892 (eos_token_id)
    
    // LLaDA-MoE特有参数
    bool qk_layernorm;       // true (qk_layernorm)
    float router_aux_loss_coef; // 0.01
    size_t dense_intermediate_size; // 8192
} LLaDAMoEMeta;


typedef struct{
    // ==================== 基础维度参数 ====================
    size_t d;                    // 2048 - 隐藏层维度
    size_t dvoc;                 // 157184 - 词表大小  
    size_t nlayer;               // 16 - Transformer层数
    size_t nh;                   // 16 - 注意力头数
    size_t nkvh;                 // 16 - KV头数
    size_t dh;                   // 128 - 每个注意力头维度 (d / nh)
    size_t di_moe;               // 1024 - 专家中间层维度
    size_t nexperts;             // 64 - 专家总数
    size_t kexperts;             // 8 - 激活专家数
    
    infiniDtype_t dt_norm, dt_mat;
    
    int transpose_linear_weights;

    // ==================== 全局权重 ====================
    // [dvoc, d]
    const void *input_embd;       // 输入词嵌入
    // [d]
    const void *output_norm;      // 输出层归一化
    // [dvoc, d] 
    const void *output_embd;      // 输出投影（可能共享输入嵌入）

    // ==================== 注意力层权重 ====================
    // nlayer * [d]
    const void *const *attn_norm; // 注意力前归一化
    
    // Q投影（两阶段低秩投影）
    // nlayer * [r_q, d] - Q第一阶段投影
    const void *const *attn_q_proj_a;
    // nlayer * [dh, r_q] - Q第二阶段投影  
    const void *const *attn_q_proj_b;
    // nlayer * [dh] - Q层归一化
    const void *const *attn_q_norm;
    
    // KV投影（两阶段低秩投影）
    // nlayer * [r_kv, d] - KV第一阶段投影
    const void *const *attn_kv_proj_a;  
    // nlayer * [2 * nkvh * dh, r_kv] - KV第二阶段投影
    const void *const *attn_kv_proj_b;
    // nlayer * [2 * nkvh * dh] - KV层归一化
    const void *const *attn_kv_norm;
    
    // 注意力输出投影
    // nlayer * [d, nh * dh]
    const void *const *attn_o_proj;

    // ==================== MoE路由权重 ====================
    // nlayer * [nexperts, d] - 路由器权重
    const void *const *moe_router_weights;
    // nlayer * [nexperts] - 路由器偏置
    const void *const *moe_router_biases;

    // ==================== 专家网络权重 ====================
    // 共享专家（如果有的话）
    // nlayer * [di_moe, d] - 共享专家gate投影
    const void *const *shared_expert_gate;
    // nlayer * [d, di_moe] - 共享专家up投影  
    const void *const *shared_expert_up;
    // nlayer * [di_moe, d] - 共享专家down投影
    const void *const *shared_expert_down;

    // MoE专家网络（每个专家独立权重）
    // nlayer * nexperts * [di_moe, d] - 专家gate投影
    const void *const *const *moe_expert_gate;
    // nlayer * nexperts * [d, di_moe] - 专家up投影
    const void *const *const *moe_expert_up;  
    // nlayer * nexperts * [di_moe, d] - 专家down投影
    const void *const *const *moe_expert_down;
} LLaDAMoEWeights;

// 改进后的权重加载器结构体
typedef struct {

} LLaDAMoEWeightLoader; // # TODO: 



__C void
createLLaDAMoEModel();

__C void
destroyLaDAMoEModel();

__C void
inferBatchLLaDAMoE();

__C void
forwardBatchLLaDAMoE();

#endif