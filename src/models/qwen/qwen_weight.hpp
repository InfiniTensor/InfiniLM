#pragma once

#include "qwen_impl.hpp"
#include "infinicore_infer.h"
#include <memory>

// 头文件中只保留函数的声明
namespace qwen {

std::shared_ptr<Tensor> getInEmbd(const QwenMeta *meta, const QwenWeights *weights);
std::shared_ptr<Tensor> getOutNorm(const QwenMeta *meta, const QwenWeights *weights);
std::shared_ptr<Tensor> getOutEmbd(const QwenMeta *meta, const QwenWeights *weights);
std::shared_ptr<Tensor> getSinTable(const QwenMeta *meta);
std::shared_ptr<Tensor> getCosTable(const QwenMeta *meta);
std::shared_ptr<Tensor> getAttnNorm(const QwenMeta *meta, const QwenWeights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnQKV(const QwenMeta *meta, const QwenWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getAttnQKVBias(const QwenMeta *meta, const QwenWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getAttnQNorm(const QwenMeta *meta, const QwenWeights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnKNorm(const QwenMeta *meta, const QwenWeights *weights, size_t layer);
std::shared_ptr<Tensor> getAttnO(const QwenMeta *meta, const QwenWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getFFNNorm(const QwenMeta *meta, const QwenWeights *weights, size_t layer);
std::shared_ptr<Tensor> getFFNGateUp(const QwenMeta *meta, const QwenWeights *weights, size_t layer, int idev, int ndev);
std::shared_ptr<Tensor> getFFNDown(const QwenMeta *meta, const QwenWeights *weights, size_t layer, int idev, int ndev);

} // namespace qwen