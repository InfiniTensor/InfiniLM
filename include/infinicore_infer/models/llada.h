#ifndef DEEPSEEK_V3_WEIGHTS_H
#define DEEPSEEK_V3_WEIGHTS_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stddef.h>
#include <stdint.h>

struct LLaDAMoEWeights;
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
    size_t n_sparse_layer;    // 16 (所有层都是MoE层) MoE 稀疏层数量
    //size_t n_dense_layer;     // 0 (没有纯稠密层)     稠密层数量
    size_t d;                 // 2048 (hidden_size) Token 向量维度
    size_t nh;                // 16 (num_attention_heads) 注意力头数
    size_t nkvh;              // 16 (num_key_value_heads) key-Value头数 和 注意力偶数一致 不是 分组查询


    size_t d_rope;            // 128 (根据partial_rotary_factor=1计算) ROPE 维度
    size_t d_nope;            // 1920 (d - d_rope)          3          不使用ROPE的维度
    size_t r_q;               // 可能需要从模型代码中获取                 Query的满秩矩阵    
    size_t r_kv;              // 可能需要从模型代码中获取                 Key-Value的秩
    size_t d_qk;              // 128 (d / nh = 2048/16)               每个注意力头的Query/Key维度（d / nh）
    size_t d_v;               // 128 (d / nh = 2048/16)。             每个注意力头的Value维度（d / nh）

    float routed_scale;       // 1.0 (routed_scaling_factor)          路由缩放因子
    size_t nexperts;          // 64 (num_experts)                     专家总数
    size_t kexperts;          // 8 (num_experts_per_tok)              每个token激活的专家数（top-k）
    size_t di;                // 2048 (hidden_size)                   输入维度（与隐藏层维度相同）
    size_t di_moe;           // 1024 (expert_intermediate_size)       专家中间层维度（专家网络瓶颈层大小）
    size_t dctx;             // 8192 (max_position_embeddings)        上下文长度（最大序列长度）
    size_t dvoc;             // 157184 (vocab_size)                   157184 - 词表大小（157,184个token）
 
    float epsilon;           // 1e-05 (rms_norm_eps)                  RMSNorm的小常数（防止除零）
    float rope_theta;        // 50000.0 (rope_theta)                  RoPE的基频（影响位置编码范围）
    uint32_t end_token;      // 156892 (eos_token_id)                 156892 - 结束标记ID
    
    // LLaDA-MoE特有参数
    bool qk_layernorm;       // true (qk_layernorm)                   对QK应用LayerNorm（提升稳定性）
    float router_aux_loss_coef; // 0.01                               路由器辅助损失系数（负载均衡）
    //size_t dense_intermediate_size; // 8192                           稠密中间层维度（如有稠密层时使用）
} LLaDAMoEConfig;

// // ----------------------------   对照表   ----------------------------------
// // 直接从config.json字段映射
// size_t d = config["hidden_size"];                    // 2048
// size_t nh = config["num_attention_heads"];           // 16  
// size_t nkvh = config["num_key_value_heads"];         // 16
// size_t nexperts = config["num_experts"];             // 64
// size_t kexperts = config["num_experts_per_tok"];     // 8
// size_t dctx = config["max_position_embeddings"];     // 8192
// size_t dvoc = config["vocab_size"];                  // 157184
// float epsilon = config["rms_norm_eps"];              // 1e-05
// float rope_theta = config["rope_theta"];             // 50000.0
// uint32_t end_token = config["eos_token_id"];         // 156892
// bool qk_layernorm = config["qk_layernorm"];          // true
// float router_aux_loss_coef = config["router_aux_loss_coef"]; // 0.01
// size_t dense_intermediate_size = config["dense_intermediate_size"]; // 8192
// float routed_scale = config["routed_scaling_factor"]; // 1.0 

// // 基于其他参数计算得出
// size_t n_sparse_layer = config["num_hidden_layers"]; // 16（因为moe_layer_freq全为1）
// size_t n_dense_layer = 0;  // 没有稠密层（因为moe_layer_freq全为1）

// size_t d_qk = config["hidden_size"] / config["num_attention_heads"]; // 2048/16=128
// size_t d_v = config["hidden_size"] / config["num_attention_heads"];  // 2048/16=128

// size_t di = config["hidden_size"];                   // 2048（输入维度=隐藏维度）
// size_t di_moe = config["expert_intermediate_size"]; // 1024

// 改进后的权重加载器结构体
typedef struct {
    // ==================== 全局权重 ====================
    load_global_fn load_token_embedding;        // 输入词嵌入（原load_input_embd）
    load_global_fn load_final_layernorm;         // 输出层归一化（原load_output_norm）
    load_global_fn load_output_projection;       // 输出投影（原load_output_embd）

    // ==================== 注意力层权重 ====================
    load_layer_fn load_attention_layernorm;      // 注意力前归一化（原load_attn_norm）
    
    // Q投影（可能的两阶段投影）
    load_layer_linear_fn load_q_proj_stage1;     // Q第一阶段投影（原load_attn_q_a_proj）
    load_layer_fn load_q_layernorm;              // Q层归一化（原load_attn_q_a_layernorm）
    load_layer_linear_fn load_q_proj_stage2;     // Q第二阶段投影（原load_attn_q_b_proj）
    
    // KV投影（可能的两阶段投影）
    load_layer_linear_fn load_kv_proj_stage1;    // KV第一阶段投影（原load_attn_kv_a_proj_with_mqa）
    load_layer_fn load_kv_layernorm;             // KV层归一化（原load_attn_kv_a_layernorm）
    load_layer_linear_fn load_kv_proj_stage2;     // KV第二阶段投影（原load_attn_kv_b_proj）
    
    load_layer_linear_fn load_output_projection; // 注意力输出投影（原load_attn_o_proj）

    // ==================== MLP层权重 ====================
    load_layer_fn load_mlp_layernorm;            // MLP前归一化（原load_mlp_norm）
    
    // 稠密MLP部分（如果有的话）
    load_layer_mlp_fn load_mlp_dense_feedforward; // 稠密前馈网络（原load_mlp_dense）

    // ==================== MoE门控路由 ====================
    load_layer_fn load_router_weights;           // 路由器权重（原load_mlp_gate_weight）
    load_layer_fn load_router_biases;            // 路由器偏置（原load_mlp_gate_bias）

    // ==================== 专家系统 ====================
    load_layer_mlp_fn load_shared_experts;       // 共享专家（原load_mlp_shared_experts）
    load_layer_expert_mlp_fn load_moe_experts;   // MoE专家网络（原load_mlp_experts）

} LLaDAWeightLoader; // # TODO: 