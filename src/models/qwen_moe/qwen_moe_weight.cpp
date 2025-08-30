#include "qwen_moe_weight.hpp"
#include <cmath>

// 假设这些类型转换辅助函数在您的项目中是可用的
// 您可能需要包含定义它们的头文件
extern uint16_t f32_to_f16(float f);
extern uint16_t f32_to_bf16(float f);

namespace qwen_moe {

// --- 全局权重加载实现 ---

std::shared_ptr<Tensor> getInEmbd(const QwenMoeMeta *meta, const QwenMoeWeights *w) {
    // 注意：MoE 模型通常不转置输入权重，但为保持一致性，我们保留这个逻辑
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
    }
}

std::shared_ptr<Tensor> getOutNorm(const QwenMoeMeta *meta, const QwenMoeWeights *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

std::shared_ptr<Tensor> getOutEmbd(const QwenMoeMeta *meta, const QwenMoeWeights *w) {
    // MoE 模型的 tie_word_embeddings 为 false，所以这是一个独立的权重
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({meta->dvoc, meta->d});
        return Tensor::weight((char *)w->output_embd, w->dt_mat, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({meta->d, meta->dvoc});
        return Tensor::weight((char *)w->output_embd, w->dt_mat, shape);
    }
}

// --- Attention 权重加载实现 (与密集模型逻辑类似) ---

std::shared_ptr<Tensor> getAttnNorm(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

std::shared_ptr<Tensor> getAttnQKV(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer, int idev, int ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * ((nh + 2 * nkvh) / ndev * dh) * d * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
        return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape);
    }
}

std::shared_ptr<Tensor> getAttnQKVBias(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer, int idev, int ndev) {
    
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    size_t offset = idev * ((nh + 2 * nkvh) / ndev * dh) * dsize(w->dt_mat);
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, w->dt_mat, shape);
}

std::shared_ptr<Tensor> getAttnQNorm(QwenMoeMeta const *meta, QwenMoeWeights const *w, size_t layer) {
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_q_norm[layer]), w->dt_norm, shape);
}

std::shared_ptr<Tensor> getAttnKNorm(QwenMoeMeta const *meta, QwenMoeWeights const *w, size_t layer) {
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_k_norm[layer]), w->dt_norm, shape);
}

std::shared_ptr<Tensor> getAttnO(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer, int idev, int ndev) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape);
    }
}

// --- MoE 专属权重加载实现 ---

std::shared_ptr<Tensor> getFFNNorm(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

std::shared_ptr<Tensor> getMoeGate(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer, int idev, int ndev) {
    auto shape = std::vector<size_t>({meta->num_experts, meta->d});
    return Tensor::weight((char *)(w->moe_gate[layer]), w->dt_mat, shape);
}

std::shared_ptr<Tensor> getMoeExpertGateUp(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer, size_t expert_idx, int idev, int ndev) {
    size_t index = layer * meta->num_experts + expert_idx;
    auto d = meta->d;
    auto moe_di = meta->moe_intermediate_size;
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({2 * moe_di, d});
        return Tensor::weight((char *)(w->moe_experts_gate_up[index]), w->dt_mat, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({d, 2 * moe_di});
        return Tensor::weight((char *)(w->moe_experts_gate_up[index]), w->dt_mat, shape);
    }
}

std::shared_ptr<Tensor> getMoeExpertDown(const QwenMoeMeta *meta, const QwenMoeWeights *w, size_t layer, size_t expert_idx, int idev, int ndev) {
    size_t index = layer * meta->num_experts + expert_idx;
    auto d = meta->d;
    auto moe_di = meta->moe_intermediate_size;
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, moe_di});
        return Tensor::weight((char *)(w->moe_experts_down[index]), w->dt_mat, shape)->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({moe_di, d});
        return Tensor::weight((char *)(w->moe_experts_down[index]), w->dt_mat, shape);
    }
}

// --- RoPE Table 生成函数 (与密集模型逻辑相同，但使用 MoE Meta) ---

std::shared_ptr<Tensor> getSinTable(const QwenMoeMeta *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);
    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float val = std::sin(static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(2*j) / meta->dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) { ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(val); }
            else if (meta->dt_logits == INFINI_DTYPE_BF16) { ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(val); }
            else if (meta->dt_logits == INFINI_DTYPE_F32) { ((float *)table)[i * half_dh + j] = val; }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

std::shared_ptr<Tensor> getCosTable(const QwenMoeMeta *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);
    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float val = std::cos(static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(2*j) / meta->dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) { ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(val); }
            else if (meta->dt_logits == INFINI_DTYPE_BF16) { ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(val); }
            else if (meta->dt_logits == INFINI_DTYPE_F32) { ((float *)table)[i * half_dh + j] = val; }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

} // namespace qwen_moe
