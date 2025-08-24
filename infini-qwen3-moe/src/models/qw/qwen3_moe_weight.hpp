#pragma once

#include "qwen3_moe_impl.hpp"
#include <memory>
#include <tensor.hpp>

// Helper function to get size of data type
inline size_t sizeof_dtype(infiniDtype_t dtype) {
    switch (dtype) {
        case INFINI_DTYPE_F16:
        case INFINI_DTYPE_BF16:
            return 2;
        case INFINI_DTYPE_F32:
        case INFINI_DTYPE_I32:
            return 4;
        case INFINI_DTYPE_F64:
        case INFINI_DTYPE_I64:
            return 8;
        default:
            return 4; // fallback
    }
}

/*
 * Qwen3-MoE权重提取工具函数
 * 用于从原始权重数据中提取适合分布式推理的权重张量
 * 
 * MoE 特性：
 * - 支持混合专家架构的权重分区
 * - 路由器权重的分布式处理
 * - 专家权重的设备间分区
 * - 稀疏激活的权重管理
 */

/*
 * 获取输入嵌入权重
 * 形状: [dvoc, d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeInEmbd(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

/*
 * 获取输出嵌入权重  
 * 形状: [d, dvoc] 或 [dvoc, d] (根据transpose_linear_weights)
 */
inline std::shared_ptr<Tensor> getQwen3MoeOutEmbd(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w) {
    if (w->transpose_linear_weights != 0) {
        // 权重存储为 [dvoc, d]，需要转置为 [d, dvoc]
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
            ->permute({1, 0});  // 转置：[dvoc, d] -> [d, dvoc]
    } else {
        // 权重已存储为 [d, dvoc]
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
    }
}

/*
 * 获取输出层归一化权重
 * 形状: [d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeOutNorm(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

/*
 * 获取注意力层归一化权重
 * 形状: [d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnNorm(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

/*
 * 获取Q归一化权重 (Qwen3特有)
 * 形状: [dh]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnQNorm(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer) {
    if (w->attn_q_norm == nullptr || w->attn_q_norm[layer] == nullptr) {
        return nullptr;  // 某些模型可能不包含Q/K归一化
    }
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_q_norm[layer]), w->dt_norm, shape);
}

/*
 * 获取K归一化权重 (Qwen3特有)
 * 形状: [dh]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnKNorm(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer) {
    if (w->attn_k_norm == nullptr || w->attn_k_norm[layer] == nullptr) {
        return nullptr;  // 某些模型可能不包含Q/K归一化
    }
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_k_norm[layer]), w->dt_norm, shape);
}

/*
 * 为分布式推理提取 Q 投影权重
 * Q 投影：[d, d] -> 每设备 [d, d/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnQ(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t output_dim_per_device = d / ndev;
    size_t offset = idev * output_dim_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        // 转置格式：[d, output_dim_per_device] -> [output_dim_per_device, d]
        auto shape = std::vector<size_t>({d, output_dim_per_device});
        return Tensor::weight((char *)(w->attn_q_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        // 标准格式：[d, output_dim_per_device]
        auto shape = std::vector<size_t>({d, output_dim_per_device});
        return Tensor::weight((char *)(w->attn_q_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 K 投影权重
 * K 投影：[d, nkvh * dh] -> 每设备 [d, (nkvh * dh)/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnK(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t nkvh = meta->nkvh;
    size_t dh = meta->dh;
    size_t kv_dim = nkvh * dh;
    size_t kv_dim_per_device = kv_dim / ndev;
    size_t offset = idev * kv_dim_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, kv_dim_per_device});
        return Tensor::weight((char *)(w->attn_k_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, kv_dim_per_device});
        return Tensor::weight((char *)(w->attn_k_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 V 投影权重
 * V 投影：[d, nkvh * dh] -> 每设备 [d, (nkvh * dh)/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnV(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t nkvh = meta->nkvh;
    size_t dh = meta->dh;
    size_t kv_dim = nkvh * dh;
    size_t kv_dim_per_device = kv_dim / ndev;
    size_t offset = idev * kv_dim_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, kv_dim_per_device});
        return Tensor::weight((char *)(w->attn_v_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, kv_dim_per_device});
        return Tensor::weight((char *)(w->attn_v_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 为分布式推理提取 O 投影权重
 * O 投影：[d, d] -> 每设备 [d/ndev, d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeAttnO(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t input_dim_per_device = d / ndev;
    size_t offset = idev * input_dim_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({input_dim_per_device, d});
        return Tensor::weight((char *)(w->attn_o_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({input_dim_per_device, d});
        return Tensor::weight((char *)(w->attn_o_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 获取MLP层归一化权重
 * 形状: [d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMLPNorm(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->mlp_norm[layer]), w->dt_norm, shape);
}

/*
 * 检查指定层是否为MoE层
 */
inline bool isQwen3MoeMoeLayer(
    Qwen3MoeMeta const *meta,
    size_t layer_idx) {
    
    // 检查是否在mlp_only_layers中
    for (size_t i = 0; i < meta->num_mlp_only_layers; ++i) {
        if (meta->mlp_only_layers[i] == layer_idx) {
            return false;  // 是纯MLP层
        }
    }
    
    // 检查是否满足MoE层的条件
    return (meta->num_experts > 0 && 
            (layer_idx + 1) % meta->decoder_sparse_step == 0);
}

/*
 * 获取普通MLP gate投影权重 (非MoE层)
 * 形状: [d, di] -> 每设备 [d, di/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMLPGate(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t di = meta->di;
    size_t di_per_device = di / ndev;
    size_t offset = idev * di_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, di_per_device});
        return Tensor::weight((char *)(w->mlp_gate_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, di_per_device});
        return Tensor::weight((char *)(w->mlp_gate_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 获取普通MLP up投影权重 (非MoE层)
 * 形状: [d, di] -> 每设备 [d, di/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMLPUp(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t di = meta->di;
    size_t di_per_device = di / ndev;
    size_t offset = idev * di_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, di_per_device});
        return Tensor::weight((char *)(w->mlp_up_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, di_per_device});
        return Tensor::weight((char *)(w->mlp_up_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 获取普通MLP down投影权重 (非MoE层)
 * 形状: [di, d] -> 每设备 [di/ndev, d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMLPDown(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t di = meta->di;
    size_t di_per_device = di / ndev;
    size_t offset = idev * di_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({di_per_device, d});
        return Tensor::weight((char *)(w->mlp_down_proj[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({di_per_device, d});
        return Tensor::weight((char *)(w->mlp_down_proj[layer]) + offset, w->dt_mat, shape);
    }
}

/*
 * 获取MoE路由器权重
 * 形状: [d, num_experts]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMoeGate(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer) {
    
    size_t d = meta->d;
    size_t num_experts = meta->num_experts;
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, num_experts});
        return Tensor::weight((char *)(w->moe_gate[layer]), w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, num_experts});
        return Tensor::weight((char *)(w->moe_gate[layer]), w->dt_mat, shape);
    }
}

/*
 * 获取MoE专家gate投影权重
 * 形状: [d, moe_intermediate_size] -> 每设备 [d, moe_intermediate_size/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMoeExpertGate(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t expert_idx, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t moe_di = meta->moe_intermediate_size;
    size_t moe_di_per_device = moe_di / ndev;
    size_t offset = idev * moe_di_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, moe_di_per_device});
        return Tensor::weight((char *)(w->moe_experts_gate_proj[layer][expert_idx]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, moe_di_per_device});
        return Tensor::weight((char *)(w->moe_experts_gate_proj[layer][expert_idx]) + offset, w->dt_mat, shape);
    }
}

/*
 * 获取MoE专家up投影权重
 * 形状: [d, moe_intermediate_size] -> 每设备 [d, moe_intermediate_size/ndev]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMoeExpertUp(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t expert_idx, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t moe_di = meta->moe_intermediate_size;
    size_t moe_di_per_device = moe_di / ndev;
    size_t offset = idev * moe_di_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, moe_di_per_device});
        return Tensor::weight((char *)(w->moe_experts_up_proj[layer][expert_idx]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, moe_di_per_device});
        return Tensor::weight((char *)(w->moe_experts_up_proj[layer][expert_idx]) + offset, w->dt_mat, shape);
    }
}

/*
 * 获取MoE专家down投影权重
 * 形状: [moe_intermediate_size, d] -> 每设备 [moe_intermediate_size/ndev, d]
 */
inline std::shared_ptr<Tensor> getQwen3MoeMoeExpertDown(
    Qwen3MoeMeta const *meta,
    Qwen3MoeWeights const *w,
    size_t layer, size_t expert_idx, size_t idev, size_t ndev) {
    
    size_t d = meta->d;
    size_t moe_di = meta->moe_intermediate_size;
    size_t moe_di_per_device = moe_di / ndev;
    size_t offset = idev * moe_di_per_device * d * sizeof_dtype(w->dt_mat);
    
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({moe_di_per_device, d});
        return Tensor::weight((char *)(w->moe_experts_down_proj[layer][expert_idx]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({moe_di_per_device, d});
        return Tensor::weight((char *)(w->moe_experts_down_proj[layer][expert_idx]) + offset, w->dt_mat, shape);
    }
}