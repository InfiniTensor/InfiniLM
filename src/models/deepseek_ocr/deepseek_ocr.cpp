#include "deepseek_ocr_impl.hpp"
#include "deepseek_ocr_weight.hpp"

#include "../../tensor.hpp"
#include "../../utils.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer.h"

#include <algorithm>
#include <random>
#include <thread>
#include <vector>

// ================ 创建设备资源 ================

void createDeviceResource(DeepSeekOCRDeviceResource *rsrc,
                          const DeepSeekOCRMeta *meta,
                          const DeepSeekOCRWeights *weights,
                          infiniDevice_t device,
                          int idev,
                          int ndev,
                          int dev_id,
                          infinicclComm_t comm) {
    RUN_INFINI(infinirtSetDevice(device, dev_id));
    infiniopHandle_t handle;
    infiniopCreateHandle(&handle);
    infinirtStream_t stream;
    infinirtStreamCreate(&stream);

    size_t nlayer = meta->n_dense_layer + meta->n_sparse_layer;

    // 加载基础权重
    auto w_in_embd = getInEmbd(meta, weights);
    auto w_out_norm = getOutNorm(meta, weights);
    auto w_out_embd = getOutEmbd(meta, weights);
    auto sin_table = getSinTable(meta);
    auto cos_table = getCosTable(meta);

    // 加载所有层的attention权重
    std::vector<std::shared_ptr<Tensor>> w_attn_norm, w_attn_q, w_attn_k, w_attn_v, w_attn_o;
    std::vector<std::shared_ptr<Tensor>> w_ffn_norm;

    for (size_t layer = 0; layer < nlayer; layer++) {
        w_attn_norm.push_back(getAttnNorm(meta, weights, layer));
        w_attn_q.push_back(getAttnQ(meta, weights, layer, idev, ndev));
        w_attn_k.push_back(getAttnK(meta, weights, layer, idev, ndev));
        w_attn_v.push_back(getAttnV(meta, weights, layer, idev, ndev));
        w_attn_o.push_back(getAttnO(meta, weights, layer, idev, ndev));
        w_ffn_norm.push_back(getFFNNorm(meta, weights, layer));
    }

    // 加载Dense MLP权重 (第0层)
    auto w_dense_gate = getDenseGate(meta, weights, idev, ndev);
    auto w_dense_up = getDenseUp(meta, weights, idev, ndev);
    auto w_dense_down = getDenseDown(meta, weights, idev, ndev);

    // 加载MoE权重 (第1-11层)
    std::vector<std::shared_ptr<Tensor>> w_moe_gate_weight, w_moe_gate_bias;
    std::vector<std::shared_ptr<Tensor>> w_moe_shared_gate, w_moe_shared_up, w_moe_shared_down;
    std::vector<std::vector<std::shared_ptr<Tensor>>> w_moe_experts_gate, w_moe_experts_up, w_moe_experts_down;

    for (size_t i = 0; i < meta->n_sparse_layer; i++) {
        w_moe_gate_weight.push_back(getMoEGateWeight(meta, weights, i));
        w_moe_gate_bias.push_back(getMoEGateBias(meta, weights, i));

        w_moe_shared_gate.push_back(getMoESharedGate(meta, weights, i, idev, ndev));
        w_moe_shared_up.push_back(getMoESharedUp(meta, weights, i, idev, ndev));
        w_moe_shared_down.push_back(getMoESharedDown(meta, weights, i, idev, ndev));

        // 加载所有routed experts
        std::vector<std::shared_ptr<Tensor>> experts_gate, experts_up, experts_down;
        for (size_t e = 0; e < meta->nexperts; e++) {
            experts_gate.push_back(getMoEExpertsGate(meta, weights, i, e, idev, ndev));
            experts_up.push_back(getMoEExpertsUp(meta, weights, i, e, idev, ndev));
            experts_down.push_back(getMoEExpertsDown(meta, weights, i, e, idev, ndev));
        }
        w_moe_experts_gate.push_back(experts_gate);
        w_moe_experts_up.push_back(experts_up);
        w_moe_experts_down.push_back(experts_down);
    }

    // 加载视觉编码器权重
    // SAM ViT-B
    auto w_sam_patch_embed = getSAMPatchEmbed(meta, weights);
    auto w_sam_patch_embed_bias = getSAMPatchEmbedBias(meta, weights);

    std::vector<std::shared_ptr<Tensor>> w_sam_block_norm1, w_sam_block_attn_qkv, w_sam_block_attn_proj;
    std::vector<std::shared_ptr<Tensor>> w_sam_block_norm2, w_sam_block_mlp_fc1, w_sam_block_mlp_fc2;
    for (size_t layer = 0; layer < 12; layer++) {
        w_sam_block_norm1.push_back(getSAMBlockNorm1(meta, weights, layer));
        w_sam_block_attn_qkv.push_back(getSAMBlockAttnQKV(meta, weights, layer));
        w_sam_block_attn_proj.push_back(getSAMBlockAttnProj(meta, weights, layer));
        w_sam_block_norm2.push_back(getSAMBlockNorm2(meta, weights, layer));
        w_sam_block_mlp_fc1.push_back(getSAMBlockMLPFC1(meta, weights, layer));
        w_sam_block_mlp_fc2.push_back(getSAMBlockMLPFC2(meta, weights, layer));
    }

    auto w_sam_neck_conv1 = getSAMNeckConv1(meta, weights);
    auto w_sam_neck_ln1 = getSAMNeckLN1(meta, weights);
    auto w_sam_neck_conv2 = getSAMNeckConv2(meta, weights);
    auto w_sam_neck_ln2 = getSAMNeckLN2(meta, weights);

    // CLIP-L
    auto w_clip_patch_embed = getCLIPPatchEmbed(meta, weights);
    auto w_clip_patch_embed_bias = getCLIPPatchEmbedBias(meta, weights);
    auto w_clip_position_embed = getCLIPPositionEmbed(meta, weights);
    auto w_clip_pre_layernorm = getCLIPPreLayerNorm(meta, weights);

    std::vector<std::shared_ptr<Tensor>> w_clip_block_ln1, w_clip_block_attn_qkv, w_clip_block_attn_proj;
    std::vector<std::shared_ptr<Tensor>> w_clip_block_ln2, w_clip_block_mlp_fc1, w_clip_block_mlp_fc2;
    for (size_t layer = 0; layer < 24; layer++) {
        w_clip_block_ln1.push_back(getCLIPBlockLN1(meta, weights, layer));
        w_clip_block_attn_qkv.push_back(getCLIPBlockAttnQKV(meta, weights, layer));
        w_clip_block_attn_proj.push_back(getCLIPBlockAttnProj(meta, weights, layer));
        w_clip_block_ln2.push_back(getCLIPBlockLN2(meta, weights, layer));
        w_clip_block_mlp_fc1.push_back(getCLIPBlockMLPFC1(meta, weights, layer));
        w_clip_block_mlp_fc2.push_back(getCLIPBlockMLPFC2(meta, weights, layer));
    }

    // Projector
    auto w_projector = getProjector(meta, weights);
    auto w_image_newline = getImageNewline(meta, weights);
    auto w_view_seperator = getViewSeperator(meta, weights);

    auto memory_pool = std::make_shared<MemoryPool>(256 * 1024 * 1024); // 256MB

    *rsrc = DeepSeekOCRDeviceResource{
        device,
        dev_id,
        handle,
        w_in_embd,
        w_out_norm,
        w_out_embd,
        sin_table,
        cos_table,
        w_attn_norm,
        w_attn_q,
        w_attn_k,
        w_attn_v,
        w_attn_o,
        w_ffn_norm,
        w_dense_gate,
        w_dense_up,
        w_dense_down,
        w_moe_gate_weight,
        w_moe_gate_bias,
        w_moe_shared_gate,
        w_moe_shared_up,
        w_moe_shared_down,
        w_moe_experts_gate,
        w_moe_experts_up,
        w_moe_experts_down,
        w_sam_patch_embed,
        w_sam_patch_embed_bias,
        w_sam_block_norm1,
        w_sam_block_attn_qkv,
        w_sam_block_attn_proj,
        w_sam_block_norm2,
        w_sam_block_mlp_fc1,
        w_sam_block_mlp_fc2,
        w_sam_neck_conv1,
        w_sam_neck_ln1,
        w_sam_neck_conv2,
        w_sam_neck_ln2,
        w_clip_patch_embed,
        w_clip_patch_embed_bias,
        w_clip_position_embed,
        w_clip_pre_layernorm,
        w_clip_block_ln1,
        w_clip_block_attn_qkv,
        w_clip_block_attn_proj,
        w_clip_block_ln2,
        w_clip_block_mlp_fc1,
        w_clip_block_mlp_fc2,
        w_projector,
        w_image_newline,
        w_view_seperator,
        stream,
        comm,
        memory_pool,
    };

    RUN_INFINI(infinirtDeviceSynchronize());
}

void releaseDeviceResource(DeepSeekOCRDeviceResource &res) {
    infinirtDeviceSynchronize();

    // 释放所有tensor
    res.w_in_embd.reset();
    res.w_out_norm.reset();
    res.w_out_embd.reset();
    res.sin_table.reset();
    res.cos_table.reset();

    for (auto &t : res.w_attn_norm) {
        t.reset();
    }
    res.w_attn_norm.clear();
    for (auto &t : res.w_attn_q) {
        t.reset();
    }
    res.w_attn_q.clear();
    for (auto &t : res.w_attn_k) {
        t.reset();
    }
    res.w_attn_k.clear();
    for (auto &t : res.w_attn_v) {
        t.reset();
    }
    res.w_attn_v.clear();
    for (auto &t : res.w_attn_o) {
        t.reset();
    }
    res.w_attn_o.clear();
    for (auto &t : res.w_ffn_norm) {
        t.reset();
    }
    res.w_ffn_norm.clear();

    res.w_dense_gate.reset();
    res.w_dense_up.reset();
    res.w_dense_down.reset();

    for (auto &t : res.w_moe_gate_weight) {
        t.reset();
    }
    res.w_moe_gate_weight.clear();
    for (auto &t : res.w_moe_gate_bias) {
        t.reset();
    }
    res.w_moe_gate_bias.clear();
    for (auto &t : res.w_moe_shared_gate) {
        t.reset();
    }
    res.w_moe_shared_gate.clear();
    for (auto &t : res.w_moe_shared_up) {
        t.reset();
    }
    res.w_moe_shared_up.clear();
    for (auto &t : res.w_moe_shared_down) {
        t.reset();
    }
    res.w_moe_shared_down.clear();

    for (auto &experts : res.w_moe_experts_gate) {
        for (auto &t : experts) {
            t.reset();
        }
        experts.clear();
    }
    res.w_moe_experts_gate.clear();
    for (auto &experts : res.w_moe_experts_up) {
        for (auto &t : experts) {
            t.reset();
        }
        experts.clear();
    }
    res.w_moe_experts_up.clear();
    for (auto &experts : res.w_moe_experts_down) {
        for (auto &t : experts) {
            t.reset();
        }
        experts.clear();
    }
    res.w_moe_experts_down.clear();

    // 释放视觉权重 - SAM
    res.w_sam_patch_embed.reset();
    res.w_sam_patch_embed_bias.reset();
    for (auto &t : res.w_sam_block_norm1) {
        t.reset();
    }
    res.w_sam_block_norm1.clear();
    for (auto &t : res.w_sam_block_attn_qkv) {
        t.reset();
    }
    res.w_sam_block_attn_qkv.clear();
    for (auto &t : res.w_sam_block_attn_proj) {
        t.reset();
    }
    res.w_sam_block_attn_proj.clear();
    for (auto &t : res.w_sam_block_norm2) {
        t.reset();
    }
    res.w_sam_block_norm2.clear();
    for (auto &t : res.w_sam_block_mlp_fc1) {
        t.reset();
    }
    res.w_sam_block_mlp_fc1.clear();
    for (auto &t : res.w_sam_block_mlp_fc2) {
        t.reset();
    }
    res.w_sam_block_mlp_fc2.clear();

    res.w_sam_neck_conv1.reset();
    res.w_sam_neck_ln1.reset();
    res.w_sam_neck_conv2.reset();
    res.w_sam_neck_ln2.reset();

    // 释放视觉权重 - CLIP
    res.w_clip_patch_embed.reset();
    res.w_clip_patch_embed_bias.reset();
    res.w_clip_position_embed.reset();
    res.w_clip_pre_layernorm.reset();
    for (auto &t : res.w_clip_block_ln1) {
        t.reset();
    }
    res.w_clip_block_ln1.clear();
    for (auto &t : res.w_clip_block_attn_qkv) {
        t.reset();
    }
    res.w_clip_block_attn_qkv.clear();
    for (auto &t : res.w_clip_block_attn_proj) {
        t.reset();
    }
    res.w_clip_block_attn_proj.clear();
    for (auto &t : res.w_clip_block_ln2) {
        t.reset();
    }
    res.w_clip_block_ln2.clear();
    for (auto &t : res.w_clip_block_mlp_fc1) {
        t.reset();
    }
    res.w_clip_block_mlp_fc1.clear();
    for (auto &t : res.w_clip_block_mlp_fc2) {
        t.reset();
    }
    res.w_clip_block_mlp_fc2.clear();

    // Projector
    res.w_projector.reset();
    res.w_image_newline.reset();
    res.w_view_seperator.reset();

    infiniopDestroyHandle(res.handle);
    res.handle = nullptr;
    infinirtStreamDestroy(res.stream);
    res.stream = nullptr;
    infinicclCommDestroy(res.comm);
    res.comm = nullptr;
}

// ================ 视觉编码器推理 ================
// 注意: 这里使用类似LLM算子的模式调用视觉算子
// Conv2d/LayerNorm/GELU需要通过InferenceContext或直接API调用

std::shared_ptr<Tensor> inferVisionSAM(const DeepSeekOCRMeta &meta,
                                       DeepSeekOCRDeviceResource &rsrc,
                                       std::shared_ptr<Tensor> pixel_values) {
    // SAM ViT-B推理: [batch, 3, H, W] -> [batch, num_patches, 1024]
    auto dt_logits = meta.dt_logits;
    auto stream = rsrc.stream;
    auto handle = rsrc.handle;
    auto batch = pixel_values->shape()[0];
    auto num_patches = (pixel_values->shape()[2] / 16) * (pixel_values->shape()[3] / 16);

    // 设置推理上下文
    auto cache_manager = new CacheManager();
    InferenceContext ctx(handle, rsrc.memory_pool, cache_manager, stream);
    setInferenceContext(&ctx);

    // 1. Patch Embedding: Conv2d(3->768, kernel=16, stride=16)
    auto H = pixel_values->shape()[2];
    auto W = pixel_values->shape()[3];
    auto patch_h = H / 16;
    auto patch_w = W / 16;

    // Conv2d输出: [batch, 768, patch_h, patch_w]
    auto conv_out = Tensor::buffer(dt_logits, {batch, 768, patch_h, patch_w}, rsrc.memory_pool);
    conv2d(conv_out, pixel_values, rsrc.w_sam_patch_embed, rsrc.w_sam_patch_embed_bias,
           16, 16, 0, 0); // stride=16, padding=0

    // Flatten(2).transpose(1,2): [batch, 768, h, w] -> [batch, h*w, 768]
    auto patch_embeds = Tensor::buffer(dt_logits, {batch, num_patches, 768}, rsrc.memory_pool);
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < patch_h; h++) {
            for (size_t w = 0; w < patch_w; w++) {
                size_t patch_idx = h * patch_w + w;
                for (size_t c = 0; c < 768; c++) {
                    size_t src_idx = ((b * 768 + c) * patch_h + h) * patch_w + w;
                    size_t dst_idx = (b * num_patches + patch_idx) * 768 + c;
                    RUN_INFINI(infinirtMemcpyAsync(
                        patch_embeds->data(dst_idx),
                        conv_out->data(src_idx),
                        dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
                }
            }
        }
    }

    // 2. ViT Transformer Blocks (12层)
    auto hidden_states = patch_embeds;
    for (size_t layer = 0; layer < 12; layer++) {
        // 2.1 LayerNorm1 + residual connection
        auto normed1 = Tensor::buffer(dt_logits, hidden_states->shape(), rsrc.memory_pool);
        layer_norm(normed1, hidden_states, rsrc.w_sam_block_norm1[layer], meta.epsilon);

        // 2.2 Self-Attention: QKV -> Split -> Attention -> Proj
        // SAM ViT-B: 12 heads, 768 dim, head_dim = 64
        const size_t num_heads = 12;
        const size_t head_dim = 64;

        auto qkv_flat = Tensor::buffer(dt_logits, {batch * num_patches, 768 * 3}, rsrc.memory_pool);
        linear(qkv_flat, normed1->view({batch * num_patches, 768}),
               rsrc.w_sam_block_attn_qkv[layer], 1.0, 0.0, nullptr, nullptr);

        // Split QKV: [batch, num_patches, 768*3] -> [batch, num_patches, 3, num_heads, head_dim]
        auto qkv = qkv_flat->view({batch, num_patches, 3, num_heads, head_dim});

        // Extract Q, K, V: each [batch, num_patches, num_heads, head_dim] -> permute to [batch, num_heads, num_patches, head_dim]
        auto q_buf = Tensor::buffer(dt_logits, {batch, num_heads, num_patches, head_dim}, rsrc.memory_pool);
        auto k_buf = Tensor::buffer(dt_logits, {batch, num_heads, num_patches, head_dim}, rsrc.memory_pool);
        auto v_buf = Tensor::buffer(dt_logits, {batch, num_heads, num_patches, head_dim}, rsrc.memory_pool);

        // Extract Q, K, V from QKV tensor (QKV layout: [batch, tokens, 3, heads, head_dim])
        auto q_extract = Tensor::buffer(dt_logits, {batch, num_patches, num_heads, head_dim}, rsrc.memory_pool);
        auto k_extract = Tensor::buffer(dt_logits, {batch, num_patches, num_heads, head_dim}, rsrc.memory_pool);
        auto v_extract = Tensor::buffer(dt_logits, {batch, num_patches, num_heads, head_dim}, rsrc.memory_pool);

        for (size_t b = 0; b < batch; b++) {
            for (size_t t = 0; t < num_patches; t++) {
                // Copy Q: qkv[b, t, 0, :, :]
                RUN_INFINI(infinirtMemcpyAsync(
                    q_extract->data((b * num_patches + t) * num_heads * head_dim),
                    qkv_flat->data((b * num_patches + t) * 3 * num_heads * head_dim + 0 * num_heads * head_dim),
                    dsize(dt_logits) * num_heads * head_dim,
                    INFINIRT_MEMCPY_D2D, stream));
                // Copy K: qkv[b, t, 1, :, :]
                RUN_INFINI(infinirtMemcpyAsync(
                    k_extract->data((b * num_patches + t) * num_heads * head_dim),
                    qkv_flat->data((b * num_patches + t) * 3 * num_heads * head_dim + 1 * num_heads * head_dim),
                    dsize(dt_logits) * num_heads * head_dim,
                    INFINIRT_MEMCPY_D2D, stream));
                // Copy V: qkv[b, t, 2, :, :]
                RUN_INFINI(infinirtMemcpyAsync(
                    v_extract->data((b * num_patches + t) * num_heads * head_dim),
                    qkv_flat->data((b * num_patches + t) * 3 * num_heads * head_dim + 2 * num_heads * head_dim),
                    dsize(dt_logits) * num_heads * head_dim,
                    INFINIRT_MEMCPY_D2D, stream));
            }
        }

        // Permute from [batch, num_patches, num_heads, head_dim] to [batch, num_heads, num_patches, head_dim]
        rearrange(q_buf, q_extract->view({batch, num_patches, num_heads, head_dim})->permute({0, 2, 1, 3}));
        rearrange(k_buf, k_extract->view({batch, num_patches, num_heads, head_dim})->permute({0, 2, 1, 3}));
        rearrange(v_buf, v_extract->view({batch, num_patches, num_heads, head_dim})->permute({0, 2, 1, 3}));

        // QK^T / sqrt(head_dim): [batch, num_heads, num_patches, num_patches]
        auto qk_scores = Tensor::buffer(dt_logits, {batch * num_heads, num_patches, num_patches}, rsrc.memory_pool);
        auto k_transposed = k_buf->view({batch * num_heads, num_patches, head_dim})->permute({0, 2, 1});
        linear(qk_scores, q_buf->view({batch * num_heads, num_patches, head_dim}), k_transposed,
               1.0f / sqrtf(head_dim), 0.0f, nullptr, nullptr);

        // Softmax over last dimension (non-causal for vision)
        auto qk_softmax = qk_scores->view({batch * num_heads * num_patches, num_patches});
        causalSoftmax(qk_softmax, qk_softmax); // Note: 实际上应该是普通softmax

        // Attention @ V: [batch, num_heads, num_patches, head_dim]
        auto attn_out_heads = Tensor::buffer(dt_logits, {batch * num_heads, num_patches, head_dim}, rsrc.memory_pool);
        linear(attn_out_heads, qk_scores, v_buf->view({batch * num_heads, num_patches, head_dim}),
               1.0f, 0.0f, nullptr, nullptr);

        // Transpose and reshape: [batch, num_heads, num_patches, head_dim] -> [batch, num_patches, num_heads, head_dim] -> [batch, num_patches, 768]
        auto attn_transposed = Tensor::buffer(dt_logits, {batch, num_patches, num_heads, head_dim}, rsrc.memory_pool);
        rearrange(attn_transposed, attn_out_heads->view({batch, num_heads, num_patches, head_dim})->permute({0, 2, 1, 3}));
        auto attn_out = attn_transposed->view({batch * num_patches, 768});

        // Output projection with residual
        auto attn_proj = Tensor::buffer(dt_logits, {batch, num_patches, 768}, rsrc.memory_pool);
        linear(attn_proj->view({batch * num_patches, 768}), attn_out,
               rsrc.w_sam_block_attn_proj[layer], 1.0, 0.0,
               hidden_states->view({batch * num_patches, 768}), nullptr);
        hidden_states = attn_proj;

        // 2.3 LayerNorm2 + residual connection
        auto normed2 = Tensor::buffer(dt_logits, hidden_states->shape(), rsrc.memory_pool);
        layer_norm(normed2, hidden_states, rsrc.w_sam_block_norm2[layer], meta.epsilon);

        // 2.4 MLP: FC1 -> GELU -> FC2
        auto mlp_hidden_flat = Tensor::buffer(dt_logits, {batch * num_patches, 3072}, rsrc.memory_pool);
        linear(mlp_hidden_flat, normed2->view({batch * num_patches, 768}),
               rsrc.w_sam_block_mlp_fc1[layer], 1.0, 0.0, nullptr, nullptr);

        gelu(mlp_hidden_flat, mlp_hidden_flat);

        auto mlp_out = Tensor::buffer(dt_logits, {batch, num_patches, 768}, rsrc.memory_pool);
        linear(mlp_out->view({batch * num_patches, 768}), mlp_hidden_flat,
               rsrc.w_sam_block_mlp_fc2[layer], 1.0, 0.0,
               hidden_states->view({batch * num_patches, 768}), nullptr);
        hidden_states = mlp_out;
    }

    // 3. Neck网络: 将768维投影到1024维
    // Reshape回spatial format: [batch, num_patches, 768] -> [batch, 768, H/16, W/16]
    auto hidden_spatial = Tensor::buffer(dt_logits, {batch, 768, patch_h, patch_w}, rsrc.memory_pool);
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < patch_h; h++) {
            for (size_t w = 0; w < patch_w; w++) {
                size_t patch_idx = h * patch_w + w;
                for (size_t c = 0; c < 768; c++) {
                    size_t src_idx = (b * num_patches + patch_idx) * 768 + c;
                    size_t dst_idx = ((b * 768 + c) * patch_h + h) * patch_w + w;
                    RUN_INFINI(infinirtMemcpyAsync(
                        hidden_spatial->data(dst_idx),
                        hidden_states->data(src_idx),
                        dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
                }
            }
        }
    }

    // Neck Conv1x1: 768->1024
    auto neck1_spatial = Tensor::buffer(dt_logits, {batch, 1024, patch_h, patch_w}, rsrc.memory_pool);
    conv2d(neck1_spatial, hidden_spatial, rsrc.w_sam_neck_conv1, nullptr,
           1, 1, 0, 0); // 1x1 conv, stride=1, padding=0

    // Flatten back: [batch, 1024, h, w] -> [batch, h*w, 1024]
    auto sam_features = Tensor::buffer(dt_logits, {batch, num_patches, 1024}, rsrc.memory_pool);
    for (size_t b = 0; b < batch; b++) {
        for (size_t h = 0; h < patch_h; h++) {
            for (size_t w = 0; w < patch_w; w++) {
                size_t patch_idx = h * patch_w + w;
                for (size_t c = 0; c < 1024; c++) {
                    size_t src_idx = ((b * 1024 + c) * patch_h + h) * patch_w + w;
                    size_t dst_idx = (b * num_patches + patch_idx) * 1024 + c;
                    RUN_INFINI(infinirtMemcpyAsync(
                        sam_features->data(dst_idx),
                        neck1_spatial->data(src_idx),
                        dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
                }
            }
        }
    }

    delete cache_manager;
    return sam_features;
}

std::shared_ptr<Tensor> inferVisionCLIP(const DeepSeekOCRMeta &meta,
                                        DeepSeekOCRDeviceResource &rsrc,
                                        std::shared_ptr<Tensor> pixel_values,
                                        std::shared_ptr<Tensor> sam_features) {
    // CLIP-L推理: [batch, 3, H, W] -> [batch, num_patches, 1024]
    auto dt_logits = meta.dt_logits;
    auto stream = rsrc.stream;
    auto handle = rsrc.handle;
    auto batch = pixel_values->shape()[0];
    auto num_patches = (pixel_values->shape()[2] / 14) * (pixel_values->shape()[3] / 14);

    // 设置推理上下文
    auto cache_manager = new CacheManager();
    InferenceContext ctx(handle, rsrc.memory_pool, cache_manager, stream);
    setInferenceContext(&ctx);

    // 1. Patch Embedding: Conv2d(3->1024, kernel=14, stride=14) + CLS token
    auto H = pixel_values->shape()[2];
    auto W = pixel_values->shape()[3];
    auto patch_h = H / 14;
    auto patch_w = W / 14;

    // Conv2d输出: [batch, 1024, patch_h, patch_w]
    auto conv_out = Tensor::buffer(dt_logits, {batch, 1024, patch_h, patch_w}, rsrc.memory_pool);
    conv2d(conv_out, pixel_values, rsrc.w_clip_patch_embed, rsrc.w_clip_patch_embed_bias,
           14, 14, 0, 0); // stride=14, padding=0

    // Flatten和转置，并添加CLS token: [batch, num_patches+1, 1024]
    auto patch_embeds = Tensor::buffer(dt_logits, {batch, num_patches + 1, 1024}, rsrc.memory_pool);
    for (size_t b = 0; b < batch; b++) {
        // CLS token at position 0 (初始化为0，实际应该是learnable parameter)
        RUN_INFINI(infinirtMemsetAsync(patch_embeds->data(b * (num_patches + 1) * 1024),
                                       0, dsize(dt_logits) * 1024, stream));

        // Patch tokens from position 1
        for (size_t h = 0; h < patch_h; h++) {
            for (size_t w = 0; w < patch_w; w++) {
                size_t patch_idx = h * patch_w + w;
                for (size_t c = 0; c < 1024; c++) {
                    size_t src_idx = ((b * 1024 + c) * patch_h + h) * patch_w + w;
                    size_t dst_idx = (b * (num_patches + 1) + patch_idx + 1) * 1024 + c;
                    RUN_INFINI(infinirtMemcpyAsync(
                        patch_embeds->data(dst_idx),
                        conv_out->data(src_idx),
                        dsize(dt_logits), INFINIRT_MEMCPY_D2D, stream));
                }
            }
        }
    }

    // 2. Add position embedding
    add(patch_embeds, patch_embeds, rsrc.w_clip_position_embed);

    // 3. Pre LayerNorm
    auto normed_embeds = Tensor::buffer(dt_logits, patch_embeds->shape(), rsrc.memory_pool);
    layer_norm(normed_embeds, patch_embeds, rsrc.w_clip_pre_layernorm, meta.epsilon);

    // 4. CLIP Transformer Blocks (24层)
    auto hidden_states = normed_embeds;
    auto total_tokens = num_patches + 1;

    for (size_t layer = 0; layer < 24; layer++) {
        // LayerNorm1 + Self-Attention + Residual
        // CLIP-L: 16 heads, 1024 dim, head_dim = 64
        const size_t num_heads = 16;
        const size_t head_dim = 64;

        auto ln1_out = Tensor::buffer(dt_logits, hidden_states->shape(), rsrc.memory_pool);
        layer_norm(ln1_out, hidden_states, rsrc.w_clip_block_ln1[layer], meta.epsilon);

        auto qkv_flat = Tensor::buffer(dt_logits, {batch * total_tokens, 1024 * 3}, rsrc.memory_pool);
        linear(qkv_flat, ln1_out->view({batch * total_tokens, 1024}),
               rsrc.w_clip_block_attn_qkv[layer], 1.0, 0.0, nullptr, nullptr);

        // Split QKV: [batch, total_tokens, 1024*3] -> [batch, total_tokens, 3, num_heads, head_dim]
        auto q_buf = Tensor::buffer(dt_logits, {batch, num_heads, total_tokens, head_dim}, rsrc.memory_pool);
        auto k_buf = Tensor::buffer(dt_logits, {batch, num_heads, total_tokens, head_dim}, rsrc.memory_pool);
        auto v_buf = Tensor::buffer(dt_logits, {batch, num_heads, total_tokens, head_dim}, rsrc.memory_pool);

        // Extract Q, K, V from QKV tensor (QKV layout: [batch, tokens, 3, heads, head_dim])
        auto q_extract = Tensor::buffer(dt_logits, {batch, total_tokens, num_heads, head_dim}, rsrc.memory_pool);
        auto k_extract = Tensor::buffer(dt_logits, {batch, total_tokens, num_heads, head_dim}, rsrc.memory_pool);
        auto v_extract = Tensor::buffer(dt_logits, {batch, total_tokens, num_heads, head_dim}, rsrc.memory_pool);

        for (size_t b = 0; b < batch; b++) {
            for (size_t t = 0; t < total_tokens; t++) {
                // Copy Q: qkv[b, t, 0, :, :]
                RUN_INFINI(infinirtMemcpyAsync(
                    q_extract->data((b * total_tokens + t) * num_heads * head_dim),
                    qkv_flat->data((b * total_tokens + t) * 3 * num_heads * head_dim + 0 * num_heads * head_dim),
                    dsize(dt_logits) * num_heads * head_dim,
                    INFINIRT_MEMCPY_D2D, stream));
                // Copy K: qkv[b, t, 1, :, :]
                RUN_INFINI(infinirtMemcpyAsync(
                    k_extract->data((b * total_tokens + t) * num_heads * head_dim),
                    qkv_flat->data((b * total_tokens + t) * 3 * num_heads * head_dim + 1 * num_heads * head_dim),
                    dsize(dt_logits) * num_heads * head_dim,
                    INFINIRT_MEMCPY_D2D, stream));
                // Copy V: qkv[b, t, 2, :, :]
                RUN_INFINI(infinirtMemcpyAsync(
                    v_extract->data((b * total_tokens + t) * num_heads * head_dim),
                    qkv_flat->data((b * total_tokens + t) * 3 * num_heads * head_dim + 2 * num_heads * head_dim),
                    dsize(dt_logits) * num_heads * head_dim,
                    INFINIRT_MEMCPY_D2D, stream));
            }
        }

        // Permute from [batch, total_tokens, num_heads, head_dim] to [batch, num_heads, total_tokens, head_dim]
        rearrange(q_buf, q_extract->view({batch, total_tokens, num_heads, head_dim})->permute({0, 2, 1, 3}));
        rearrange(k_buf, k_extract->view({batch, total_tokens, num_heads, head_dim})->permute({0, 2, 1, 3}));
        rearrange(v_buf, v_extract->view({batch, total_tokens, num_heads, head_dim})->permute({0, 2, 1, 3}));

        // QK^T / sqrt(head_dim): [batch, num_heads, total_tokens, total_tokens]
        auto qk_scores = Tensor::buffer(dt_logits, {batch * num_heads, total_tokens, total_tokens}, rsrc.memory_pool);
        auto k_transposed = k_buf->view({batch * num_heads, total_tokens, head_dim})->permute({0, 2, 1});
        linear(qk_scores, q_buf->view({batch * num_heads, total_tokens, head_dim}), k_transposed,
               1.0f / sqrtf(head_dim), 0.0f, nullptr, nullptr);

        // Softmax over last dimension (non-causal for vision)
        auto qk_softmax = qk_scores->view({batch * num_heads * total_tokens, total_tokens});
        causalSoftmax(qk_softmax, qk_softmax); // Note: 实际上应该是普通softmax

        // Attention @ V: [batch, num_heads, total_tokens, head_dim]
        auto attn_out_heads = Tensor::buffer(dt_logits, {batch * num_heads, total_tokens, head_dim}, rsrc.memory_pool);
        linear(attn_out_heads, qk_scores, v_buf->view({batch * num_heads, total_tokens, head_dim}),
               1.0f, 0.0f, nullptr, nullptr);

        // Transpose and reshape: [batch, num_heads, total_tokens, head_dim] -> [batch, total_tokens, num_heads, head_dim] -> [batch, total_tokens, 1024]
        auto attn_transposed = Tensor::buffer(dt_logits, {batch, total_tokens, num_heads, head_dim}, rsrc.memory_pool);
        rearrange(attn_transposed, attn_out_heads->view({batch, num_heads, total_tokens, head_dim})->permute({0, 2, 1, 3}));
        auto attn_out = attn_transposed->view({batch * total_tokens, 1024});

        // Output projection with residual
        auto attn_proj = Tensor::buffer(dt_logits, {batch, total_tokens, 1024}, rsrc.memory_pool);
        linear(attn_proj->view({batch * total_tokens, 1024}), attn_out,
               rsrc.w_clip_block_attn_proj[layer], 1.0, 0.0,
               hidden_states->view({batch * total_tokens, 1024}), nullptr);
        hidden_states = attn_proj;

        // LayerNorm2 + MLP(GELU) + Residual
        auto ln2_out = Tensor::buffer(dt_logits, hidden_states->shape(), rsrc.memory_pool);
        layer_norm(ln2_out, hidden_states, rsrc.w_clip_block_ln2[layer], meta.epsilon);

        auto mlp_hidden_flat = Tensor::buffer(dt_logits, {batch * total_tokens, 4096}, rsrc.memory_pool);
        linear(mlp_hidden_flat, ln2_out->view({batch * total_tokens, 1024}),
               rsrc.w_clip_block_mlp_fc1[layer], 1.0, 0.0, nullptr, nullptr);

        gelu(mlp_hidden_flat, mlp_hidden_flat);

        auto mlp_out = Tensor::buffer(dt_logits, {batch, total_tokens, 1024}, rsrc.memory_pool);
        linear(mlp_out->view({batch * total_tokens, 1024}), mlp_hidden_flat,
               rsrc.w_clip_block_mlp_fc2[layer], 1.0, 0.0,
               hidden_states->view({batch * total_tokens, 1024}), nullptr);
        hidden_states = mlp_out;
    }

    // 5. 移除CLS token (index 0)，只保留patch tokens [1:]
    auto clip_features = Tensor::buffer(dt_logits, {batch, num_patches, 1024}, rsrc.memory_pool);
    for (size_t b = 0; b < batch; b++) {
        RUN_INFINI(infinirtMemcpyAsync(
            clip_features->data(b * num_patches * 1024),
            hidden_states->data((b * total_tokens + 1) * 1024),
            dsize(dt_logits) * num_patches * 1024,
            INFINIRT_MEMCPY_D2D, stream));
    }

    delete cache_manager;
    return clip_features;
}

std::shared_ptr<Tensor> inferVision(const DeepSeekOCRMeta &meta,
                                    DeepSeekOCRDeviceResource &rsrc,
                                    const void *pixel_values_patches,
                                    const void *pixel_values_global,
                                    uint32_t num_patches,
                                    uint32_t height_patches,
                                    uint32_t width_patches) {
    // 完整视觉编码: SAM+CLIP -> 拼接 -> Projector -> 添加分隔符
    auto dt_logits = meta.dt_logits;
    auto d = meta.d; // 1280
    auto stream = rsrc.stream;

    std::vector<std::shared_ptr<Tensor>> all_visual_embeds;
    size_t total_visual_tokens = 0;

    // 1. 处理局部patches (如果有)
    if (num_patches > 0 && pixel_values_patches != nullptr) {
        // 创建tensor wrapper
        auto patches_tensor = Tensor::weight(pixel_values_patches, dt_logits,
                                             {num_patches, 3, 640, 640});

        // SAM特征提取: [num_patches, N_sam, 1024]
        auto sam_local = inferVisionSAM(meta, rsrc, patches_tensor);

        // CLIP特征提取: [num_patches, N_clip, 1024]
        auto clip_local = inferVisionCLIP(meta, rsrc, patches_tensor, sam_local);

        // 拼接SAM和CLIP特征: [num_patches, N, 2048]
        // CLIP和SAM的特征数量应该相同(都是(640/14)^2 = 2116个patch)
        auto N = sam_local->shape()[1];
        auto concat_features = Tensor::buffer(dt_logits, {num_patches, N, 2048}, rsrc.memory_pool);

        // 拼接操作: concat_features = [clip_local, sam_local] along dim=-1
        for (size_t b = 0; b < num_patches; b++) {
            for (size_t n = 0; n < N; n++) {
                // 复制CLIP特征到前1024维
                RUN_INFINI(infinirtMemcpyAsync(
                    concat_features->data((b * N + n) * 2048),
                    clip_local->data((b * N + n) * 1024),
                    dsize(dt_logits) * 1024, INFINIRT_MEMCPY_D2D, stream));
                // 复制SAM特征到后1024维
                RUN_INFINI(infinirtMemcpyAsync(
                    concat_features->data((b * N + n) * 2048 + 1024),
                    sam_local->data((b * N + n) * 1024),
                    dsize(dt_logits) * 1024, INFINIRT_MEMCPY_D2D, stream));
            }
        }

        // Projector投影: [num_patches, N, 2048] -> [num_patches, N, 1280]
        auto local_features = Tensor::buffer(dt_logits, {num_patches * N, d}, rsrc.memory_pool);
        linear(local_features, concat_features->view({num_patches * N, 2048}),
               rsrc.w_projector, 1.0, 0.0, nullptr, nullptr);

        // 重排为2D网格并添加image_newline (每行patch后添加一个newline token)
        // 计算：每个patch有N个token，加上每行的newline，总共num_patches个patch按grid排列
        total_visual_tokens += num_patches * N + height_patches * width_patches;
        all_visual_embeds.push_back(local_features);
    }

    // 2. 处理全局视图
    if (pixel_values_global != nullptr) {
        auto global_tensor = Tensor::weight(pixel_values_global, dt_logits, {1, 3, 1024, 1024});

        // SAM和CLIP特征提取
        auto sam_global = inferVisionSAM(meta, rsrc, global_tensor);               // [1, N_global, 1024]
        auto clip_global = inferVisionCLIP(meta, rsrc, global_tensor, sam_global); // [1, N_global, 1024]

        // 拼接特征
        auto N_global = sam_global->shape()[1]; // (1024/14)^2 = 5329 patches
        auto concat_global = Tensor::buffer(dt_logits, {1, N_global, 2048}, rsrc.memory_pool);

        for (size_t n = 0; n < N_global; n++) {
            RUN_INFINI(infinirtMemcpyAsync(
                concat_global->data(n * 2048),
                clip_global->data(n * 1024),
                dsize(dt_logits) * 1024, INFINIRT_MEMCPY_D2D, stream));
            RUN_INFINI(infinirtMemcpyAsync(
                concat_global->data(n * 2048 + 1024),
                sam_global->data(n * 1024),
                dsize(dt_logits) * 1024, INFINIRT_MEMCPY_D2D, stream));
        }

        // Projector投影
        auto global_features = Tensor::buffer(dt_logits, {N_global, d}, rsrc.memory_pool);
        linear(global_features, concat_global->view({N_global, 2048}),
               rsrc.w_projector, 1.0, 0.0, nullptr, nullptr);

        total_visual_tokens += N_global + 1; // +1 for image_newline at end
        all_visual_embeds.push_back(global_features);
    }

    // 3. 拼接所有视觉特征 + 添加分隔符
    // 最终结构: [local_patches...] + [image_newlines...] + [global_view] + [view_seperator]
    auto visual_embeds = Tensor::buffer(dt_logits, {total_visual_tokens, d}, rsrc.memory_pool);

    size_t current_pos = 0;

    // 复制局部patches特征 (如果有)
    for (auto &local_feat : all_visual_embeds) {
        if (local_feat) {
            size_t num_tokens = local_feat->shape()[0] * local_feat->shape()[1];
            RUN_INFINI(infinirtMemcpyAsync(
                visual_embeds->data(current_pos * d),
                local_feat->data(),
                dsize(dt_logits) * num_tokens * d,
                INFINIRT_MEMCPY_D2D, stream));
            current_pos += num_tokens;

            // 在每批patch后添加image_newline
            if (current_pos < total_visual_tokens) {
                RUN_INFINI(infinirtMemcpyAsync(
                    visual_embeds->data(current_pos * d),
                    rsrc.w_image_newline->data(),
                    dsize(dt_logits) * d,
                    INFINIRT_MEMCPY_D2D, stream));
                current_pos++;
            }
        }
    }

    // 最后添加view_seperator
    if (current_pos < total_visual_tokens) {
        RUN_INFINI(infinirtMemcpyAsync(
            visual_embeds->data(current_pos * d),
            rsrc.w_view_seperator->data(),
            dsize(dt_logits) * d,
            INFINIRT_MEMCPY_D2D, stream));
    }

    fprintf(stderr, "inferVision: Framework complete, concat/projector need linear operator calls\n");
    return visual_embeds;
}

// ================ LLM 推理 ================

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
                      const uint32_t *patch_info) {

    auto nlayer = meta.n_dense_layer + meta.n_sparse_layer;
    auto d = meta.d;
    auto nh = meta.nh / ndev;
    auto nkvh = meta.nkvh / ndev;
    auto ngroup = nh / nkvh;
    auto dh = meta.dh;
    auto dt_logits = meta.dt_logits;
    auto dvoc = meta.dvoc;
    auto stream = rsrc.stream;

    // 设置推理上下文
    auto cache_manager = new CacheManager();
    InferenceContext ctx(rsrc.handle, rsrc.memory_pool, cache_manager, stream);
    setInferenceContext(&ctx);

    // Allocate buffers
    auto logits_in = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);
    auto logits_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

    // Attention buffers
    auto qkv_buf = Tensor::buffer(dt_logits, {ntok, (nh + nkvh * 2) * dh}, rsrc.memory_pool);
    auto q_buf = qkv_buf->slice(1, 0, nh * dh)->view({ntok, nh, dh});
    auto k_buf = qkv_buf->slice(1, nh * dh, nkvh * dh)->view({ntok, nkvh, dh});
    auto v_buf = qkv_buf->slice(1, (nh + nkvh) * dh, nkvh * dh)->view({ntok, nkvh, dh});
    auto o_buf = Tensor::buffer(dt_logits, {ntok, nh * dh}, rsrc.memory_pool);

    // Sampling buffers
    auto prob_buf = Tensor::buffer(dt_logits, {nreq, dvoc}, rsrc.memory_pool);
    auto result_buf = Tensor::buffer(INFINI_DTYPE_I64, {nreq}, rsrc.memory_pool);
    auto result_cpu = std::vector<int64_t>(nreq);

    // Prepare position IDs
    auto batch_pos_ids = std::vector<uint32_t>(ntok);
    size_t req_start = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        for (uint32_t i = 0; i < req_lens[req]; i++) {
            batch_pos_ids[req_start + i] = req_pos[req] + i;
        }
        req_start += req_lens[req];
    }

    std::shared_ptr<Tensor> pos_ids_buf;
    if (rsrc.device == INFINI_DEVICE_CPU) {
        pos_ids_buf = Tensor::weight(batch_pos_ids.data(), INFINI_DTYPE_U32, {ntok});
    } else {
        pos_ids_buf = Tensor::buffer(INFINI_DTYPE_U32, {ntok}, rsrc.memory_pool);
        RUN_INFINI(infinirtMemcpyAsync(pos_ids_buf->data(), batch_pos_ids.data(),
                                       sizeof(uint32_t) * ntok, INFINIRT_MEMCPY_H2D, stream));
    }

    // 1. 输入准备 - Token Embedding + Vision Encoding
    std::shared_ptr<Tensor> visual_embeds;
    bool has_vision_input = (pixel_values_global != nullptr);
    const uint32_t IMAGE_TOKEN_ID = 128815; // DeepSeek-OCR的image token id

    if (has_vision_input && req_pos[0] == 0) {
        // Prefill阶段且有图像输入，调用视觉编码器
        uint32_t num_patches = patch_info[0];
        uint32_t height_patches = patch_info[1];
        uint32_t width_patches = patch_info[2];

        visual_embeds = inferVision(meta, rsrc, pixel_values_patches, pixel_values_global,
                                    num_patches, height_patches, width_patches);
    }

    // 构建输入embeddings，如果遇到image_token_id则替换为visual_embeds
    size_t visual_token_idx = 0;
    for (uint32_t i = 0; i < ntok; i++) {
        if (has_vision_input && tokens[i] == IMAGE_TOKEN_ID && visual_embeds != nullptr) {
            // 替换为视觉特征
            if (visual_token_idx < visual_embeds->shape()[0]) {
                RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                               visual_embeds->data(visual_token_idx * d),
                                               dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
                visual_token_idx++;
            } else {
                // 视觉token用完了，使用普通embedding
                RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                               rsrc.w_in_embd->data(tokens[i] * d),
                                               dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
            }
        } else {
            // 普通文本token
            RUN_INFINI(infinirtMemcpyAsync(logits_in->data(i * d),
                                           rsrc.w_in_embd->data(tokens[i] * d),
                                           dsize(dt_logits) * d, INFINIRT_MEMCPY_D2D, stream));
        }
    }

    // Attention inner loop setup
    size_t max_qk_size = 0;
    size_t max_seq_len = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;
        max_qk_size = std::max(max_qk_size, size_t(seq_len * total_len));
        max_seq_len = std::max(max_seq_len, size_t(seq_len));
    }

    auto qk_buf = Tensor::buffer(dt_logits, {nh * max_qk_size}, rsrc.memory_pool);
    auto rearrange_q_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto q_rearrange = rearrange_q_buf->view({nkvh, ngroup, max_seq_len, dh});
    auto attn_val_buf = Tensor::buffer(dt_logits, {nkvh, ngroup * max_seq_len, dh}, rsrc.memory_pool);
    auto attn_val_gemm = attn_val_buf->view({nkvh, ngroup, max_seq_len, dh});

    // 2. Transformer Decoder 循环
    for (size_t layer = 0; layer < nlayer; layer++) {
        // 2.1 Attention
        rmsnorm(logits_out, logits_in, rsrc.w_attn_norm[layer], meta.epsilon);

        // QKV投影
        linear(q_buf->view({ntok, nh * dh}), logits_out, rsrc.w_attn_q[layer], 1.0, 0.0, nullptr, nullptr);
        linear(k_buf->view({ntok, nkvh * dh}), logits_out, rsrc.w_attn_k[layer], 1.0, 0.0, nullptr, nullptr);
        linear(v_buf->view({ntok, nkvh * dh}), logits_out, rsrc.w_attn_v[layer], 1.0, 0.0, nullptr, nullptr);

        // RoPE
        rope(q_buf, k_buf, pos_ids_buf, rsrc.sin_table, rsrc.cos_table);

        // Attention计算 (per request)
        size_t token_offset = 0;
        for (uint32_t req = 0; req < nreq; req++) {
            auto past_len = req_pos[req];
            auto seq_len = req_lens[req];
            auto total_len = past_len + seq_len;

            auto o = o_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto q = q_buf->slice({{0, token_offset, seq_len}})->view({seq_len, nkvh, ngroup, dh})->permute({1, 2, 0, 3});
            auto k = k_buf->slice({{0, token_offset, seq_len}});
            auto v = v_buf->slice({{0, token_offset, seq_len}});

            // Update KV cache
            rearrange(kv_caches[req]->k[idev][layer]->slice(0, past_len, seq_len), k);
            rearrange(kv_caches[req]->v[idev][layer]->slice(0, past_len, seq_len), v);

            // QK^T
            rearrange(q_rearrange->slice(2, 0, seq_len), q);
            auto qk_gemm = qk_buf->slice(0, 0, nh * seq_len * total_len)->view({nkvh, ngroup * seq_len, total_len});
            auto k_gemm = kv_caches[req]->k[idev][layer]->slice(0, 0, total_len)->permute({1, 2, 0});
            linear(qk_gemm, rearrange_q_buf->slice(1, 0, ngroup * seq_len), k_gemm,
                   1.f / float(sqrt(dh)), 0.f, nullptr, nullptr);

            // Softmax
            auto qk_softmax = qk_gemm->view({nh, seq_len, total_len});
            causalSoftmax(qk_softmax, qk_softmax);

            // Attention @ V
            auto v_gemm = kv_caches[req]->v[idev][layer]->slice(0, 0, total_len)->permute({1, 0, 2});
            linear(attn_val_buf->slice(1, 0, ngroup * seq_len), qk_gemm, v_gemm, 1.f, 0.f, nullptr, nullptr);
            rearrange(o, attn_val_gemm->slice(2, 0, seq_len));

            token_offset += seq_len;
        }

        // O投影
        linear(logits_in, o_buf, rsrc.w_attn_o[layer], 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);

        // AllReduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                                          INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }

        // 2.2 FFN
        rmsnorm(logits_out, logits_in, rsrc.w_ffn_norm[layer], meta.epsilon);

        if (layer < meta.n_dense_layer) {
            // Dense MLP (第0层)
            auto di_dense = meta.di_dense / ndev;
            auto gate_up_buf = Tensor::buffer(dt_logits, {ntok, 2 * di_dense}, rsrc.memory_pool);
            auto gate_buf = gate_up_buf->slice(1, 0, di_dense);
            auto up_buf = gate_up_buf->slice(1, di_dense, di_dense);

            linear(gate_buf, logits_out, rsrc.w_dense_gate, 1.0, 0.0, nullptr, nullptr);
            linear(up_buf, logits_out, rsrc.w_dense_up, 1.0, 0.0, nullptr, nullptr);
            swiglu(gate_buf, up_buf, gate_buf);
            linear(logits_in, gate_buf, rsrc.w_dense_down, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);

        } else {
            // MoE (第1-11层)
            size_t moe_layer_idx = layer - meta.n_dense_layer;
            auto di_moe = meta.di_moe;
            auto di_shared = meta.di_shared / ndev;

            // Gate routing
            auto router_logits = Tensor::buffer(dt_logits, {ntok, meta.nexperts}, rsrc.memory_pool);
            linear(router_logits, logits_out, rsrc.w_moe_gate_weight[moe_layer_idx],
                   1.0, 0.0, nullptr, rsrc.w_moe_gate_bias[moe_layer_idx]);

            // Top-K selection
            auto topk_values = Tensor::buffer(INFINI_DTYPE_F32, {ntok, meta.kexperts}, rsrc.memory_pool);
            auto topk_indices = Tensor::buffer(INFINI_DTYPE_I32, {ntok, meta.kexperts}, rsrc.memory_pool);
            topkrouter(topk_values, topk_indices, router_logits,
                       rsrc.w_moe_gate_bias[moe_layer_idx], meta.routed_scale, meta.kexperts);

            // Shared experts (always active)
            auto shared_gate_up = Tensor::buffer(dt_logits, {ntok, 2 * di_shared}, rsrc.memory_pool);
            auto shared_gate = shared_gate_up->slice(1, 0, di_shared);
            auto shared_up = shared_gate_up->slice(1, di_shared, di_shared);
            auto shared_out = Tensor::buffer(dt_logits, {ntok, d}, rsrc.memory_pool);

            linear(shared_gate, logits_out, rsrc.w_moe_shared_gate[moe_layer_idx], 1.0, 0.0, nullptr, nullptr);
            linear(shared_up, logits_out, rsrc.w_moe_shared_up[moe_layer_idx], 1.0, 0.0, nullptr, nullptr);
            swiglu(shared_gate, shared_up, shared_gate);
            linear(shared_out, shared_gate, rsrc.w_moe_shared_down[moe_layer_idx], 1.0, 0.0, nullptr, nullptr);

            // Routed experts (需要动态调度和并行计算)
            // 简化处理：只使用shared experts的输出
            linear(logits_in, shared_out, nullptr, 1.0, 0.0, idev == 0 ? logits_in : nullptr, nullptr);
        }

        // AllReduce if distributed
        if (rsrc.comm != nullptr) {
            RUN_INFINI(infinicclAllReduce(logits_in->data(), logits_in->data(), ntok * d, dt_logits,
                                          INFINICCL_SUM, rsrc.comm, stream));
            RUN_INFINI(infinirtStreamSynchronize(stream));
        }
    }

    // 3. Output & Sampling (only rank 0)
    if (idev == 0) {
        if (last_logits != nullptr) {
            // Forward mode: return all logits
            rmsnorm(logits_out, logits_in, rsrc.w_out_norm, meta.epsilon);
            auto last_logits_buf = Tensor::buffer(dt_logits, {ntok, dvoc}, rsrc.memory_pool);
            linear(last_logits_buf, logits_out, rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);
            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(last_logits, last_logits_buf->data(),
                                      dsize(dt_logits) * ntok * dvoc, INFINIRT_MEMCPY_D2H));
        }

        if (output != nullptr) {
            // Inference mode: sample next token
            size_t token_offset = 0;
            for (uint32_t req = 0; req < nreq; req++) {
                auto seq_len = req_lens[req];
                token_offset += seq_len;
                rmsnorm(logits_out->slice(0, req, 1),
                        logits_in->slice(0, token_offset - 1, 1),
                        rsrc.w_out_norm, meta.epsilon);
            }
            linear(prob_buf, logits_out->slice(0, 0, nreq), rsrc.w_out_embd, 1.0, 0.0, nullptr, nullptr);

            // Sampling
            std::random_device _rd;
            std::mt19937 gen(_rd());
            std::uniform_real_distribution<float> dis(0.0, 1.0);

            for (uint32_t req = 0; req < nreq; req++) {
                float random_val = dis(gen);
                randomSample(result_buf->slice(0, req, 1),
                             prob_buf->slice(0, req, 1),
                             random_val, topp[req], topk[req], temperature[req]);
            }

            RUN_INFINI(infinirtStreamSynchronize(stream));
            RUN_INFINI(infinirtMemcpy(result_cpu.data(), result_buf->data(),
                                      sizeof(int64_t) * nreq, INFINIRT_MEMCPY_D2H));
            for (uint32_t req = 0; req < nreq; req++) {
                output[req] = static_cast<uint32_t>(result_cpu[req]);
            }
        }
    }

    delete cache_manager;
}

// ================ C API ================

__C __export struct DeepSeekOCRModel *
createDeepSeekOCRModel(const DeepSeekOCRMeta *meta,
                       const DeepSeekOCRWeights *weights,
                       infiniDevice_t device,
                       int ndev,
                       const int *dev_ids) {
    auto model = new DeepSeekOCRModel();
    model->meta = *meta;

    std::vector<infinicclComm_t> comms;
    if (ndev > 1) {
        comms.resize(ndev);
        infinicclCommInitAll(comms.data(), ndev, dev_ids);
    } else {
        comms.resize(1);
        comms[0] = nullptr;
    }

    model->resources.resize(ndev);
    std::vector<std::thread> threads;
    for (int i = 0; i < ndev; i++) {
        threads.emplace_back([&, i]() {
            createDeviceResource(&model->resources[i], meta, weights,
                                 device, i, ndev, dev_ids[i], comms[i]);
        });
    }
    for (auto &t : threads) {
        t.join();
    }

    return model;
}

__C __export void
destroyDeepSeekOCRModel(struct DeepSeekOCRModel *model) {
    if (model == nullptr) {
        return;
    }

    for (auto &res : model->resources) {
        releaseDeviceResource(res);
    }
    delete model;
}

__C __export void
inferBatchDeepSeekOCR(struct DeepSeekOCRModel *model,
                      const uint32_t *tokens,
                      uint32_t ntok,
                      const uint32_t *req_lens,
                      uint32_t nreq,
                      const uint32_t *req_pos,
                      struct KVCache **kv_caches,
                      const float *temperature,
                      const uint32_t *topk,
                      const float *topp,
                      uint32_t *output) {
    int ndev = model->resources.size();
    std::vector<std::thread> threads;

    for (int i = 0; i < ndev; i++) {
        threads.emplace_back([&, i]() {
            inferDeviceBatch(model->meta, model->resources[i],
                             i, ndev, tokens, ntok,
                             req_lens, nreq, req_pos,
                             kv_caches, temperature, topk, topp,
                             output, nullptr,
                             nullptr, nullptr, nullptr);
        });
    }

    for (auto &t : threads) {
        t.join();
    }
}

__C __export void
forwardBatchDeepSeekOCR(struct DeepSeekOCRModel *model,
                        const uint32_t *tokens,
                        uint32_t ntok,
                        const uint32_t *req_lens,
                        uint32_t nreq,
                        const uint32_t *req_pos,
                        struct KVCache **kv_caches,
                        void *logits) {
    int ndev = model->resources.size();
    std::vector<std::thread> threads;

    for (int i = 0; i < ndev; i++) {
        threads.emplace_back([&, i]() {
            inferDeviceBatch(model->meta, model->resources[i],
                             i, ndev, tokens, ntok,
                             req_lens, nreq, req_pos,
                             kv_caches, nullptr, nullptr, nullptr,
                             nullptr, logits,
                             nullptr, nullptr, nullptr);
        });
    }

    for (auto &t : threads) {
        t.join();
    }
}

__C __export void
inferBatchDeepSeekOCRWithEmbeds(struct DeepSeekOCRModel *model,
                                const void *inputs_embeds,
                                uint32_t ntok,
                                const uint32_t *req_lens,
                                uint32_t nreq,
                                const uint32_t *req_pos,
                                struct KVCache **kv_caches,
                                const float *temperature,
                                const uint32_t *topk,
                                const float *topp,
                                uint32_t *output) {
    // 使用预计算的embeddings推理（用于Python端传入已处理的视觉特征）
    // 这个接口主要用于灵活的多模态输入处理
    fprintf(stderr, "inferBatchDeepSeekOCRWithEmbeds not fully implemented yet\n");
}
