#ifndef DEEPSEEK_OCR_IMPL_HPP
#define DEEPSEEK_OCR_IMPL_HPP

#include "../../../allocator.hpp"
#include "../../../cache.hpp"
#include "../../../tensor.hpp"
#include "infinicore_infer/cache.h"
#include "infinicore_infer/models/deepseek_ocr.h"

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <memory>
#include <vector>

// 设备资源结构
struct DeepSeekOCRDeviceResource {
    infiniDevice_t device;
    int dev_id;
    infiniopHandle_t handle;

    // 基础权重
    std::shared_ptr<Tensor> w_in_embd;
    std::shared_ptr<Tensor> w_out_norm;
    std::shared_ptr<Tensor> w_out_embd;

    // RoPE表
    std::shared_ptr<Tensor> sin_table;
    std::shared_ptr<Tensor> cos_table;

    // Attention权重 (所有层)
    std::vector<std::shared_ptr<Tensor>> w_attn_norm;
    std::vector<std::shared_ptr<Tensor>> w_attn_q;
    std::vector<std::shared_ptr<Tensor>> w_attn_k;
    std::vector<std::shared_ptr<Tensor>> w_attn_v;
    std::vector<std::shared_ptr<Tensor>> w_attn_o;

    // FFN norm (所有层)
    std::vector<std::shared_ptr<Tensor>> w_ffn_norm;

    // Dense MLP权重 (第0层)
    std::shared_ptr<Tensor> w_dense_gate;
    std::shared_ptr<Tensor> w_dense_up;
    std::shared_ptr<Tensor> w_dense_down;

    // MoE权重 (第1-11层)
    std::vector<std::shared_ptr<Tensor>> w_moe_gate_weight;
    std::vector<std::shared_ptr<Tensor>> w_moe_gate_bias;

    // Shared experts
    std::vector<std::shared_ptr<Tensor>> w_moe_shared_gate;
    std::vector<std::shared_ptr<Tensor>> w_moe_shared_up;
    std::vector<std::shared_ptr<Tensor>> w_moe_shared_down;

    // Routed experts (n_sparse_layer * nexperts)
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_gate;
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_up;
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_down;

    // Vision Encoder weights
    // SAM ViT-B (12 layers)
    std::shared_ptr<Tensor> w_sam_patch_embed; // Conv2d weight
    std::shared_ptr<Tensor> w_sam_patch_embed_bias;
    std::vector<std::shared_ptr<Tensor>> w_sam_block_norm1; // 12 layers
    std::vector<std::shared_ptr<Tensor>> w_sam_block_attn_qkv;
    std::vector<std::shared_ptr<Tensor>> w_sam_block_attn_proj;
    std::vector<std::shared_ptr<Tensor>> w_sam_block_norm2;
    std::vector<std::shared_ptr<Tensor>> w_sam_block_mlp_fc1;
    std::vector<std::shared_ptr<Tensor>> w_sam_block_mlp_fc2;
    std::shared_ptr<Tensor> w_sam_neck_conv1; // Neck conv layers
    std::shared_ptr<Tensor> w_sam_neck_ln1;
    std::shared_ptr<Tensor> w_sam_neck_conv2;
    std::shared_ptr<Tensor> w_sam_neck_ln2;

    // CLIP-L (24 layers)
    std::shared_ptr<Tensor> w_clip_patch_embed; // Conv2d weight
    std::shared_ptr<Tensor> w_clip_patch_embed_bias;
    std::shared_ptr<Tensor> w_clip_position_embed;
    std::shared_ptr<Tensor> w_clip_pre_layernorm;
    std::vector<std::shared_ptr<Tensor>> w_clip_block_ln1; // 24 layers
    std::vector<std::shared_ptr<Tensor>> w_clip_block_attn_qkv;
    std::vector<std::shared_ptr<Tensor>> w_clip_block_attn_proj;
    std::vector<std::shared_ptr<Tensor>> w_clip_block_ln2;
    std::vector<std::shared_ptr<Tensor>> w_clip_block_mlp_fc1;
    std::vector<std::shared_ptr<Tensor>> w_clip_block_mlp_fc2;

    // Projector
    std::shared_ptr<Tensor> w_projector;      // [2048, 1280]
    std::shared_ptr<Tensor> w_image_newline;  // [1280]
    std::shared_ptr<Tensor> w_view_seperator; // [1280]

    infinirtStream_t stream;
    infinicclComm_t comm;
    std::shared_ptr<MemoryPool> memory_pool;
};

// 模型结构
struct DeepSeekOCRModel {
    DeepSeekOCRMeta meta;
    std::vector<DeepSeekOCRDeviceResource> resources;
};

// 创建设备资源
void createDeviceResource(DeepSeekOCRDeviceResource *rsrc,
                          const DeepSeekOCRMeta *meta,
                          const DeepSeekOCRWeights *weights,
                          infiniDevice_t device,
                          int idev,
                          int ndev,
                          int dev_id,
                          infinicclComm_t comm);

// 释放设备资源
void releaseDeviceResource(DeepSeekOCRDeviceResource &res);

// SAM ViT-B 视觉编码器推理
std::shared_ptr<Tensor> inferVisionSAM(const DeepSeekOCRMeta &meta,
                                       DeepSeekOCRDeviceResource &rsrc,
                                       std::shared_ptr<Tensor> pixel_values);

// CLIP-L 视觉编码器推理
std::shared_ptr<Tensor> inferVisionCLIP(const DeepSeekOCRMeta &meta,
                                        DeepSeekOCRDeviceResource &rsrc,
                                        std::shared_ptr<Tensor> pixel_values,
                                        std::shared_ptr<Tensor> sam_features);

// 完整视觉编码（SAM + CLIP + Projector）
std::shared_ptr<Tensor> inferVision(const DeepSeekOCRMeta &meta,
                                    DeepSeekOCRDeviceResource &rsrc,
                                    const void *pixel_values_patches,
                                    const void *pixel_values_global,
                                    uint32_t num_patches,
                                    uint32_t height_patches,
                                    uint32_t width_patches);

// 在单个设备上进行推理
void inferDeviceBatch(const DeepSeekOCRMeta &meta,
                      DeepSeekOCRDeviceResource &rsrc,
                      uint32_t idev,
                      uint32_t ndev,
                      const uint32_t *tokens,
                      uint32_t ntok,
                      const uint32_t *req_lens,
                      uint32_t nreq,
                      const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature,
                      const uint32_t *topk,
                      const float *topp,
                      uint32_t *output,
                      void *last_logits,
                      const void *pixel_values_patches,
                      const void *pixel_values_global,
                      const uint32_t *patch_info);

#endif // DEEPSEEK_OCR_IMPL_HPP
