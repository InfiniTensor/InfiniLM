#ifndef DEEPSEEK_OCR_WEIGHT_HPP
#define DEEPSEEK_OCR_WEIGHT_HPP

#include "../../../tensor.hpp"
#include "infinicore_infer/models/deepseek_ocr.h"

#include <cmath>
#include <memory>

// 获取输入embedding
inline std::shared_ptr<Tensor> getInEmbd(const DeepSeekOCRMeta *meta,
                                         const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->input_embd, weights->dt_mat,
                          {meta->dvoc, meta->d},
                          weights->transpose_linear_weights);
}

// 获取输出norm
inline std::shared_ptr<Tensor> getOutNorm(const DeepSeekOCRMeta *meta,
                                          const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->output_norm, weights->dt_norm, {meta->d});
}

// 获取输出embedding
inline std::shared_ptr<Tensor> getOutEmbd(const DeepSeekOCRMeta *meta,
                                          const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->output_embd, weights->dt_mat,
                          {meta->dvoc, meta->d},
                          weights->transpose_linear_weights);
}

// 获取RoPE sin表
inline std::shared_ptr<Tensor> getSinTable(const DeepSeekOCRMeta *meta) {
    size_t dctx = meta->dctx;
    size_t dh = meta->dh;
    float theta = meta->theta;

    std::vector<float> sin_table(dctx * dh);
    for (size_t pos = 0; pos < dctx; ++pos) {
        for (size_t i = 0; i < dh; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / dh);
            sin_table[pos * dh + i] = std::sin(pos * freq);
        }
    }
    return Tensor::weight(sin_table.data(), INFINI_DTYPE_F32, {dctx, dh});
}

// 获取RoPE cos表
inline std::shared_ptr<Tensor> getCosTable(const DeepSeekOCRMeta *meta) {
    size_t dctx = meta->dctx;
    size_t dh = meta->dh;
    float theta = meta->theta;

    std::vector<float> cos_table(dctx * dh);
    for (size_t pos = 0; pos < dctx; ++pos) {
        for (size_t i = 0; i < dh; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / dh);
            cos_table[pos * dh + i] = std::cos(pos * freq);
        }
    }
    return Tensor::weight(cos_table.data(), INFINI_DTYPE_F32, {dctx, dh});
}

// ===================== Attention Weights =====================

inline std::shared_ptr<Tensor> getAttnNorm(const DeepSeekOCRMeta *meta,
                                           const DeepSeekOCRWeights *weights,
                                           size_t layer) {
    return Tensor::weight(weights->attn_norm[layer], weights->dt_norm, {meta->d});
}

inline std::shared_ptr<Tensor> getAttnQ(const DeepSeekOCRMeta *meta,
                                        const DeepSeekOCRWeights *weights,
                                        size_t layer,
                                        int idev,
                                        int ndev) {
    size_t d = meta->d;
    size_t nh = meta->nh;
    size_t dh = meta->dh;
    size_t nh_per_dev = nh / ndev;

    return Tensor::weight(weights->attn_q[layer], weights->dt_mat,
                          {nh_per_dev * dh, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getAttnK(const DeepSeekOCRMeta *meta,
                                        const DeepSeekOCRWeights *weights,
                                        size_t layer,
                                        int idev,
                                        int ndev) {
    size_t d = meta->d;
    size_t nkvh = meta->nkvh;
    size_t dh = meta->dh;
    size_t nkvh_per_dev = nkvh / ndev;

    return Tensor::weight(weights->attn_k[layer], weights->dt_mat,
                          {nkvh_per_dev * dh, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getAttnV(const DeepSeekOCRMeta *meta,
                                        const DeepSeekOCRWeights *weights,
                                        size_t layer,
                                        int idev,
                                        int ndev) {
    size_t d = meta->d;
    size_t nkvh = meta->nkvh;
    size_t dh = meta->dh;
    size_t nkvh_per_dev = nkvh / ndev;

    return Tensor::weight(weights->attn_v[layer], weights->dt_mat,
                          {nkvh_per_dev * dh, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getAttnO(const DeepSeekOCRMeta *meta,
                                        const DeepSeekOCRWeights *weights,
                                        size_t layer,
                                        int idev,
                                        int ndev) {
    size_t d = meta->d;
    size_t nh = meta->nh;
    size_t dh = meta->dh;
    size_t nh_per_dev = nh / ndev;

    return Tensor::weight(weights->attn_o[layer], weights->dt_mat,
                          {d, nh_per_dev * dh},
                          weights->transpose_linear_weights);
}

// ===================== FFN Weights =====================

inline std::shared_ptr<Tensor> getFFNNorm(const DeepSeekOCRMeta *meta,
                                          const DeepSeekOCRWeights *weights,
                                          size_t layer) {
    return Tensor::weight(weights->ffn_norm[layer], weights->dt_norm, {meta->d});
}

// ===================== Dense MLP Weights (Layer 0) =====================

inline std::shared_ptr<Tensor> getDenseGate(const DeepSeekOCRMeta *meta,
                                            const DeepSeekOCRWeights *weights,
                                            int idev,
                                            int ndev) {
    size_t d = meta->d;
    size_t di_dense = meta->di_dense;
    size_t di_per_dev = di_dense / ndev;

    return Tensor::weight(weights->dense_gate, weights->dt_mat,
                          {di_per_dev, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getDenseUp(const DeepSeekOCRMeta *meta,
                                          const DeepSeekOCRWeights *weights,
                                          int idev,
                                          int ndev) {
    size_t d = meta->d;
    size_t di_dense = meta->di_dense;
    size_t di_per_dev = di_dense / ndev;

    return Tensor::weight(weights->dense_up, weights->dt_mat,
                          {di_per_dev, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getDenseDown(const DeepSeekOCRMeta *meta,
                                            const DeepSeekOCRWeights *weights,
                                            int idev,
                                            int ndev) {
    size_t d = meta->d;
    size_t di_dense = meta->di_dense;
    size_t di_per_dev = di_dense / ndev;

    return Tensor::weight(weights->dense_down, weights->dt_mat,
                          {d, di_per_dev},
                          weights->transpose_linear_weights);
}

// ===================== MoE Weights (Layer 1-11) =====================

inline std::shared_ptr<Tensor> getMoEGateWeight(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights,
                                                size_t sparse_layer_idx) {
    size_t d = meta->d;
    size_t nexperts = meta->nexperts;
    return Tensor::weight(weights->moe_gate_weight[sparse_layer_idx],
                          weights->dt_mat,
                          {nexperts, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getMoEGateBias(const DeepSeekOCRMeta *meta,
                                              const DeepSeekOCRWeights *weights,
                                              size_t sparse_layer_idx) {
    size_t nexperts = meta->nexperts;
    return Tensor::weight(weights->moe_gate_bias[sparse_layer_idx],
                          weights->dt_mat,
                          {nexperts});
}

// Shared experts
inline std::shared_ptr<Tensor> getMoESharedGate(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights,
                                                size_t sparse_layer_idx,
                                                int idev,
                                                int ndev) {
    size_t d = meta->d;
    size_t di_shared = meta->di_shared;
    size_t di_per_dev = di_shared / ndev;

    return Tensor::weight(weights->moe_shared_gate[sparse_layer_idx],
                          weights->dt_mat,
                          {di_per_dev, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getMoESharedUp(const DeepSeekOCRMeta *meta,
                                              const DeepSeekOCRWeights *weights,
                                              size_t sparse_layer_idx,
                                              int idev,
                                              int ndev) {
    size_t d = meta->d;
    size_t di_shared = meta->di_shared;
    size_t di_per_dev = di_shared / ndev;

    return Tensor::weight(weights->moe_shared_up[sparse_layer_idx],
                          weights->dt_mat,
                          {di_per_dev, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getMoESharedDown(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights,
                                                size_t sparse_layer_idx,
                                                int idev,
                                                int ndev) {
    size_t d = meta->d;
    size_t di_shared = meta->di_shared;
    size_t di_per_dev = di_shared / ndev;

    return Tensor::weight(weights->moe_shared_down[sparse_layer_idx],
                          weights->dt_mat,
                          {d, di_per_dev},
                          weights->transpose_linear_weights);
}

// Routed experts
inline std::shared_ptr<Tensor> getMoEExpertsGate(const DeepSeekOCRMeta *meta,
                                                 const DeepSeekOCRWeights *weights,
                                                 size_t sparse_layer_idx,
                                                 size_t expert_idx,
                                                 int idev,
                                                 int ndev) {
    size_t d = meta->d;
    size_t di_moe = meta->di_moe;

    return Tensor::weight(weights->moe_experts_gate[sparse_layer_idx][expert_idx],
                          weights->dt_mat,
                          {di_moe, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getMoEExpertsUp(const DeepSeekOCRMeta *meta,
                                               const DeepSeekOCRWeights *weights,
                                               size_t sparse_layer_idx,
                                               size_t expert_idx,
                                               int idev,
                                               int ndev) {
    size_t d = meta->d;
    size_t di_moe = meta->di_moe;

    return Tensor::weight(weights->moe_experts_up[sparse_layer_idx][expert_idx],
                          weights->dt_mat,
                          {di_moe, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getMoEExpertsDown(const DeepSeekOCRMeta *meta,
                                                 const DeepSeekOCRWeights *weights,
                                                 size_t sparse_layer_idx,
                                                 size_t expert_idx,
                                                 int idev,
                                                 int ndev) {
    size_t d = meta->d;
    size_t di_moe = meta->di_moe;

    return Tensor::weight(weights->moe_experts_down[sparse_layer_idx][expert_idx],
                          weights->dt_mat,
                          {d, di_moe},
                          weights->transpose_linear_weights);
}

// ===================== Vision Weights =====================

// SAM ViT-B weights
inline std::shared_ptr<Tensor> getSAMPatchEmbed(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->sam_patch_embed, weights->dt_mat,
                          {768, 3, 16, 16}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMPatchEmbedBias(const DeepSeekOCRMeta *meta,
                                                    const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->sam_patch_embed_bias, weights->dt_mat, {768});
}

inline std::shared_ptr<Tensor> getSAMBlockNorm1(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights,
                                                size_t layer) {
    return Tensor::weight(weights->sam_block_norm1[layer], weights->dt_norm, {768});
}

inline std::shared_ptr<Tensor> getSAMBlockAttnQKV(const DeepSeekOCRMeta *meta,
                                                  const DeepSeekOCRWeights *weights,
                                                  size_t layer) {
    return Tensor::weight(weights->sam_block_attn_qkv[layer], weights->dt_mat,
                          {768 * 3, 768}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMBlockAttnProj(const DeepSeekOCRMeta *meta,
                                                   const DeepSeekOCRWeights *weights,
                                                   size_t layer) {
    return Tensor::weight(weights->sam_block_attn_proj[layer], weights->dt_mat,
                          {768, 768}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMBlockNorm2(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights,
                                                size_t layer) {
    return Tensor::weight(weights->sam_block_norm2[layer], weights->dt_norm, {768});
}

inline std::shared_ptr<Tensor> getSAMBlockMLPFC1(const DeepSeekOCRMeta *meta,
                                                 const DeepSeekOCRWeights *weights,
                                                 size_t layer) {
    return Tensor::weight(weights->sam_block_mlp_fc1[layer], weights->dt_mat,
                          {3072, 768}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMBlockMLPFC2(const DeepSeekOCRMeta *meta,
                                                 const DeepSeekOCRWeights *weights,
                                                 size_t layer) {
    return Tensor::weight(weights->sam_block_mlp_fc2[layer], weights->dt_mat,
                          {768, 3072}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMNeckConv1(const DeepSeekOCRMeta *meta,
                                               const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->sam_neck_conv1, weights->dt_mat,
                          {1024, 768, 1, 1}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMNeckLN1(const DeepSeekOCRMeta *meta,
                                             const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->sam_neck_ln1, weights->dt_norm, {1024});
}

inline std::shared_ptr<Tensor> getSAMNeckConv2(const DeepSeekOCRMeta *meta,
                                               const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->sam_neck_conv2, weights->dt_mat,
                          {1024, 1024, 3, 3}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getSAMNeckLN2(const DeepSeekOCRMeta *meta,
                                             const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->sam_neck_ln2, weights->dt_norm, {1024});
}

// CLIP-L weights
inline std::shared_ptr<Tensor> getCLIPPatchEmbed(const DeepSeekOCRMeta *meta,
                                                 const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->clip_patch_embed, weights->dt_mat,
                          {1024, 3, 14, 14}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getCLIPPatchEmbedBias(const DeepSeekOCRMeta *meta,
                                                     const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->clip_patch_embed_bias, weights->dt_mat, {1024});
}

inline std::shared_ptr<Tensor> getCLIPPositionEmbed(const DeepSeekOCRMeta *meta,
                                                    const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->clip_position_embed, weights->dt_mat, {257, 1024});
}

inline std::shared_ptr<Tensor> getCLIPPreLayerNorm(const DeepSeekOCRMeta *meta,
                                                   const DeepSeekOCRWeights *weights) {
    return Tensor::weight(weights->clip_pre_layernorm, weights->dt_norm, {1024});
}

inline std::shared_ptr<Tensor> getCLIPBlockLN1(const DeepSeekOCRMeta *meta,
                                               const DeepSeekOCRWeights *weights,
                                               size_t layer) {
    return Tensor::weight(weights->clip_block_ln1[layer], weights->dt_norm, {1024});
}

inline std::shared_ptr<Tensor> getCLIPBlockAttnQKV(const DeepSeekOCRMeta *meta,
                                                   const DeepSeekOCRWeights *weights,
                                                   size_t layer) {
    return Tensor::weight(weights->clip_block_attn_qkv[layer], weights->dt_mat,
                          {1024 * 3, 1024}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getCLIPBlockAttnProj(const DeepSeekOCRMeta *meta,
                                                    const DeepSeekOCRWeights *weights,
                                                    size_t layer) {
    return Tensor::weight(weights->clip_block_attn_proj[layer], weights->dt_mat,
                          {1024, 1024}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getCLIPBlockLN2(const DeepSeekOCRMeta *meta,
                                               const DeepSeekOCRWeights *weights,
                                               size_t layer) {
    return Tensor::weight(weights->clip_block_ln2[layer], weights->dt_norm, {1024});
}

inline std::shared_ptr<Tensor> getCLIPBlockMLPFC1(const DeepSeekOCRMeta *meta,
                                                  const DeepSeekOCRWeights *weights,
                                                  size_t layer) {
    return Tensor::weight(weights->clip_block_mlp_fc1[layer], weights->dt_mat,
                          {4096, 1024}, weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getCLIPBlockMLPFC2(const DeepSeekOCRMeta *meta,
                                                  const DeepSeekOCRWeights *weights,
                                                  size_t layer) {
    return Tensor::weight(weights->clip_block_mlp_fc2[layer], weights->dt_mat,
                          {1024, 4096}, weights->transpose_linear_weights);
}

// Projector
inline std::shared_ptr<Tensor> getProjector(const DeepSeekOCRMeta *meta,
                                            const DeepSeekOCRWeights *weights) {
    size_t d = meta->d; // 1280
    return Tensor::weight(weights->projector, weights->dt_mat,
                          {2048, d},
                          weights->transpose_linear_weights);
}

inline std::shared_ptr<Tensor> getImageNewline(const DeepSeekOCRMeta *meta,
                                               const DeepSeekOCRWeights *weights) {
    size_t d = meta->d;
    return Tensor::weight(weights->image_newline, weights->dt_mat, {d});
}

inline std::shared_ptr<Tensor> getViewSeperator(const DeepSeekOCRMeta *meta,
                                                const DeepSeekOCRWeights *weights) {
    size_t d = meta->d;
    return Tensor::weight(weights->view_seperator, weights->dt_mat, {d});
}

#endif // DEEPSEEK_OCR_WEIGHT_HPP
