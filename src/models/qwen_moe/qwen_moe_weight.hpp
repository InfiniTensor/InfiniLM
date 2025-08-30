#ifndef QWEN_MOE_WEIGHT_H
#define QWEN_MOE_WEIGHT_H

#include "infinicore_infer/models/qwen_moe.h" // 使用 MoE 的头文件
#include "../../tensor.hpp"
#include <memory>

// 使用一个独立的命名空间，以避免与密集模型发生冲突
namespace qwen_moe {

// --- MoE 权重加载函数的声明 ---

// 全局权重
std::shared_ptr<Tensor> getInEmbd(const QwenMoeMeta *meta, const QwenMoeWeights *weights);
std::shared_ptr<Tensor> getOutNorm(const QwenMoeMeta *meta, const QwenMoeWeights *weights);
std::shared_ptr<Tensor> getOutEmbd(const QwenMoeMeta *meta, const QwenMoeWeights *weights);
std::shared_ptr<Tensor> getSinTable(const QwenMoeMeta *meta);
std::shared_ptr<Tensor> getCosTable(const QwenMoeMeta *meta);

// 逐层 Attention 权重
std::shared_ptr<Tensor> getAttnNorm(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnQKV(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getAttnQKVBias(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getAttnQNorm(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnKNorm(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnO(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer, int idev, int ndev);

// 逐层 MoE 专属权重
std::shared_ptr<Tensor> getFFNNorm(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer);
std::shared_ptr<Tensor> getMoeGate(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getMoeExpertGateUp(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer, size_t expert_idx, int idev, int ndev);
std::shared_ptr<Tensor> getMoeExpertDown(const QwenMoeMeta *meta, const QwenMoeWeights *weights, size_t layer, size_t expert_idx, int idev, int ndev);

} // namespace qwen_moe

#endif // QWEN_MOE_WEIGHT_H
