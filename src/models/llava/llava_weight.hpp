#ifndef LLAVA_WEIGHT_HPP
#define LLAVA_WEIGHT_HPP

#include "llava_impl.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer/models/llava.h"

#include <memory>

// // Vision weight getters
// inline std::shared_ptr<Tensor> getPatchEmbedWeight(const LlavaWeights *weights) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{},  // Shape to be defined based on vision encoder
//         weights->vision_patch_embed_weight,
//         DT_F16, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getProjectorWeight(const LlavaWeights *weights, size_t text_dim, size_t vision_dim) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{text_dim, vision_dim},
//         weights->projector_weight,
//         DT_F16, DEVICE_CPU, 0
//     );
// }

// // Reuse Jiuge weight getters for language model
// inline std::shared_ptr<Tensor> getAttnNorm(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{meta->d},
//         weights->attn_norm[layer],
//         meta->dt_norm, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getAttnQKV(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer, int idev, int ndev) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{(meta->nh + 2 * meta->nkvh) / ndev, meta->dh, meta->d},
//         weights->attn_qkv[layer],
//         meta->dt_mat, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getAttnQKVBias(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer, int idev, int ndev) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{(meta->nh + 2 * meta->nkvh) / ndev, meta->dh},
//         weights->attn_qkv_b[layer],
//         meta->dt_mat, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getAttnQNorm(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{meta->dh},
//         weights->attn_q_norm[layer],
//         meta->dt_norm, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getAttnKNorm(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{meta->dh},
//         weights->attn_k_norm[layer],
//         meta->dt_norm, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getAttnO(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer, int idev, int ndev) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{meta->d, meta->nkvh / ndev * meta->dh},
//         weights->attn_o[layer],
//         meta->dt_mat, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getFFNNorm(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{meta->d},
//         weights->ffn_norm[layer],
//         meta->dt_norm, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getFFNGateUp(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer, int idev, int ndev) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{2 * meta->di / ndev, meta->d},
//         weights->ffn_gate_up[layer],
//         meta->dt_mat, DEVICE_CPU, 0
//     );
// }

// inline std::shared_ptr<Tensor> getFFNDown(const LlavaLanguageMeta *meta, const LlavaWeights *weights, size_t layer, int idev, int ndev) {
//     return std::make_shared<Tensor>(
//         std::vector<size_t>{meta->d, meta->di / ndev},
//         weights->ffn_down[layer],
//         meta->dt_mat, DEVICE_CPU, 0
//     );
// }

#endif