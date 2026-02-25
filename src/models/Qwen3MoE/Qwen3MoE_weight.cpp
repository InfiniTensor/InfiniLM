#include "Qwen3MoE_impl.hpp"
#include "infinicore_infer.h"

#include "../../tensor.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <iostream>

// ==================== 辅助函数 ====================

// 辅助函数：创建普通线性权重 (BF16)
// 形状通常为 [in_dim, out_dim]，这是 InfiniLM 计算库的标准格式
inline std::shared_ptr<Tensor> getLinear(
    const Qwen3MoEAttentionMeta *meta, size_t in_dim, size_t out_dim) {
    // 创建 BF16 权重张量
    auto shape = std::vector<size_t>({in_dim, out_dim});
    // 使用 meta->dtype 也可以，通常 meta->dtype 已经是 BF16
    return Tensor::weight(nullptr, INFINI_DTYPE_BF16, shape);
}

// 辅助函数：分布式加载线性权重 (Tensor Parallel)
// 即使 ndev=1 也能正常工作
inline void load_dist_linear(void *w_ptr, std::shared_ptr<Tensor> w, 
                             size_t ndev, size_t dev, infinirtStream_t stream) {
    // 简单假设按输出维度切分 (Column Parallel)
    // 偏移量 = 总元素数 / ndev * dev * 元素大小
    size_t offset = w->shape()[0] * w->shape()[1] * dev * dsize(w->dtype());
    w->load(reinterpret_cast<char *>(w_ptr) + offset, stream);
}

// 获取Attention Norm权重
inline std::shared_ptr<Tensor> getAttnNorm(const Qwen3MoEAttentionMeta *meta) {
    auto shape = std::vector<size_t>({meta->hidden_size});
    return Tensor::weight(nullptr, meta->dtype, shape);
}

// 1. 恢复标准 Sin/Cos 表 (适用于 64 dim -> 32 freqs)
inline std::shared_ptr<Tensor> getSinTable(const Qwen3MoEAttentionMeta *meta) {
    auto half_dh = meta->head_dim / 2; 
    auto unit = dsize(meta->dtype);
    void *table = std::malloc(meta->max_seq_len * half_dh * unit);
    float theta = meta->rope_theta;

    // 标准 Full RoPE 生成逻辑
    for (size_t i = 0; i < meta->max_seq_len; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            // j = 0..31
            float freq_exponent = static_cast<float>(j) / static_cast<float>(half_dh);
            float freq = std::pow(theta, freq_exponent);
            float _sin = std::sin(static_cast<float>(i) / freq);
            
            size_t idx = i * half_dh + j;
            if (meta->dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[idx] = f32_to_bf16(_sin);
            } else if (meta->dtype == INFINI_DTYPE_F32) {
                ((float *)table)[idx] = _sin;
            }
        }
    }
    // ... (Tensor 创建代码同上)
    auto shape = std::vector<size_t>({meta->max_seq_len, half_dh});
    auto tensor = Tensor::weight(table, meta->dtype, shape);
    std::free(table);
    return tensor;
}

// Cos 表同理，完全标准逻辑
inline std::shared_ptr<Tensor> getCosTable(const Qwen3MoEAttentionMeta *meta) {
    auto half_dh = meta->head_dim / 2;
    auto unit = dsize(meta->dtype);
    void *table = std::malloc(meta->max_seq_len * half_dh * unit);
    float theta = meta->rope_theta;

    // 标准 Full RoPE 生成逻辑
    for (size_t i = 0; i < meta->max_seq_len; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            // j = 0..31
            float freq_exponent = static_cast<float>(j) / static_cast<float>(half_dh);
            float freq = std::pow(theta, freq_exponent);
            float _cos = std::cos(static_cast<float>(i) / freq);
            
            size_t idx = i * half_dh + j;
            if (meta->dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[idx] = f32_to_bf16(_cos);
            } else if (meta->dtype == INFINI_DTYPE_F32) {
                ((float *)table)[idx] = _cos;
            }
        }
    }
    auto shape = std::vector<size_t>({meta->max_seq_len, half_dh});
    auto tensor = Tensor::weight(table, meta->dtype, shape);
    std::free(table);
    return tensor;
}

// 恢复 Norm 权重形状
inline std::shared_ptr<Tensor> getQNorm(const Qwen3MoEAttentionMeta *meta) {
    auto shape = std::vector<size_t>({meta->head_dim}); // 128
    return Tensor::weight(nullptr, meta->dtype, shape);
}

inline std::shared_ptr<Tensor> getKNorm(const Qwen3MoEAttentionMeta *meta) {
    //std::cout<<"head dim"<<meta->head_dim<<std::endl;
    auto shape = std::vector<size_t>({meta->head_dim}); // 128
    return Tensor::weight(nullptr, meta->dtype, shape);
}
// ==================== 构造函数 ====================

Qwen3MoEWeights::Qwen3MoEWeights(
    const Qwen3MoEAttentionMeta *meta,
    infiniDevice_t device,
    int ndev,
    const int *dev_ids) {
    
    device_weights = std::vector<std::shared_ptr<Qwen3DeviceWeights>>(ndev);
    
    // 假设只有1层attention
    size_t nlayer = 1;
    
    // 计算本地头数 (Tensor Parallel)
    size_t local_num_heads = meta->num_heads / ndev;
    size_t local_num_kv_heads = meta->num_kv_head / ndev;
    
    for (int dev = 0; dev < ndev; dev++) {
        int dev_id = dev_ids[dev];
        RUN_INFINI(infinirtSetDevice(device, dev_id));
        device_weights[dev] = std::make_shared<Qwen3DeviceWeights>();
        device_weights[dev]->device = device;
        device_weights[dev]->dev_id = dev_id;
        RUN_INFINI(infinirtStreamCreate(&device_weights[dev]->load_stream));

        // 初始化RoPE表
        device_weights[dev]->sin_table = getSinTable(meta);
        device_weights[dev]->cos_table = getCosTable(meta);

        // 初始化layers
        device_weights[dev]->w_layers = std::vector<Qwen3LayerWeight>(nlayer);
        
        for (size_t layer = 0; layer < nlayer; layer++) {
            auto attn_weight = std::make_shared<Qwen3AttentionWeight>();
            
            // Pre-Norm
            attn_weight->attn_norm = getAttnNorm(meta);
            
            // Q/K/V投影（GQA + Tensor Parallel）
            // 注意：这里 out_dim 使用本地头数计算
            size_t q_out_dim = local_num_heads * meta->head_dim;
            size_t kv_out_dim = local_num_kv_heads * meta->head_dim;
            
            // 【修改点】改为使用 getLinear 初始化普通 BF16 Tensor
            attn_weight->q_proj = getLinear(meta, meta->hidden_size, q_out_dim);
            attn_weight->k_proj = getLinear(meta, meta->hidden_size, kv_out_dim);
            attn_weight->v_proj = getLinear(meta, meta->hidden_size, kv_out_dim);
            
            // QK Norm
            attn_weight->q_norm = getQNorm(meta); 
            attn_weight->k_norm = getKNorm(meta);
            
            // Output投影
            // 注意：Output Proj 输入维度切分，输出维度完整 (Row Parallel 归约)
            // 这里为了简化加载逻辑，我们暂时假设它也是普通 Linear
            attn_weight->o_proj = getLinear(meta, q_out_dim, meta->hidden_size);
            
            device_weights[dev]->w_layers[layer].self_attn = attn_weight;
        }
    }
}

// ==================== 权重加载函数 (移除 Scale/Zero 参数) ====================

// 加载Attention Norm
void load_attn_norm(Qwen3MoEWeights *weights, void *cpu_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer_id].self_attn->attn_norm->load(cpu_ptr, weight->load_stream);
    }
}

// 加载Q投影
// 【修改点】去掉了 scale_ptr, zero_ptr
void load_attn_q_proj(Qwen3MoEWeights *weights, void *weight_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        
        auto q_proj = weight->w_layers[layer_id].self_attn->q_proj;
        load_dist_linear(weight_ptr, q_proj, weights->device_weights.size(), dev, weight->load_stream);
    }
}

// 加载K投影
void load_attn_k_proj(Qwen3MoEWeights *weights, void *weight_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        
        auto k_proj = weight->w_layers[layer_id].self_attn->k_proj;
        load_dist_linear(weight_ptr, k_proj, weights->device_weights.size(), dev, weight->load_stream);
    }
}

// 加载V投影
void load_attn_v_proj(Qwen3MoEWeights *weights, void *weight_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        
        auto v_proj = weight->w_layers[layer_id].self_attn->v_proj;
        load_dist_linear(weight_ptr, v_proj, weights->device_weights.size(), dev, weight->load_stream);
    }
}

// 加载Q Norm
void load_attn_q_norm(Qwen3MoEWeights *weights, void *cpu_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer_id].self_attn->q_norm->load(cpu_ptr, weight->load_stream);
    }
}

// 加载K Norm
void load_attn_k_norm(Qwen3MoEWeights *weights, void *cpu_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        weight->w_layers[layer_id].self_attn->k_norm->load(cpu_ptr, weight->load_stream);
    }
}

// 加载Output投影
void load_attn_o_proj(Qwen3MoEWeights *weights, void *weight_ptr, size_t layer_id) {
    for (int dev = 0; dev < int(weights->device_weights.size()); dev++) {
        auto weight = weights->device_weights[dev];
        RUN_INFINI(infinirtSetDevice(weight->device, weight->dev_id));
        
        auto o_proj = weight->w_layers[layer_id].self_attn->o_proj;
        load_dist_linear(weight_ptr, o_proj, weights->device_weights.size(), dev, weight->load_stream);
    }
}

// 创建权重加载器
// 【修改点】结构体定义需要去对应修改头文件，这里只填入函数指针
static Qwen3MoEWeightLoader weight_loader = {
    .load_attn_norm = load_attn_norm,
    .load_attn_q_proj = load_attn_q_proj,
    .load_attn_k_proj = load_attn_k_proj,
    .load_attn_v_proj = load_attn_v_proj,
    .load_attn_q_norm = load_attn_q_norm,
    .load_attn_k_norm = load_attn_k_norm,
    .load_attn_o_proj = load_attn_o_proj,
};

__C __export Qwen3MoEWeights *
createQwen3MoEWeights(const Qwen3MoEAttentionMeta *meta,
                      infiniDevice_t device,
                      int ndev,
                      const int *dev_ids) {
    auto weights = new Qwen3MoEWeights(meta, device, ndev, dev_ids);
    return weights;
}

__C __export Qwen3MoEWeightLoader *
createQwen3MoEWeightLoader() {
    return &weight_loader;
}