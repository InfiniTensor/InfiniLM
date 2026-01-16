#ifndef LLAVA_WEIGHT_HPP
#define LLAVA_WEIGHT_HPP

#include "llava_impl.hpp"
#include "../inference_context.hpp"
#include "infinicore_infer/models/llava.h"

#include <memory>
#include <cstring>  // for memcpy
#include <cstdlib>

inline bool llava_debug_enabled_in_weighthpp() {
    static int cached = -1;
    if (cached == -1) {
        const char *env = std::getenv("LLAVA_DEBUG");
        cached = (env != nullptr && std::strcmp(env, "0") != 0) ? 1 : 0;
    }
    return cached != 0;
}

// Vision weight getters
inline std::shared_ptr<Tensor> getPatchEmbedWeight(
    LlavaMeta const *meta,
    LlavaWeights const *weights) {
    // 从meta中获取vision embedding参数
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;  // 输出通道数 [1024]
    auto patch_size = meta->vision_meta.patch_size;              // 卷积核大小 [14]

    // 对于RGB图像，输入通道数总是3
    const size_t input_channels = 3;

    // Patch embedding卷积核形状: [vision_embed_dim, input_channels, patch_size, patch_size]
    auto shape = std::vector<size_t>{vision_embed_dim, input_channels, patch_size, patch_size};

    if (llava_debug_enabled_in_weighthpp()) {
        printf("[CPP getPatchEmbedWeight] vision_patch_embed_weight pointer: %p\n", weights->vision_patch_embed_weight);
    }
    auto vision_patch_embed_device_tensor =
    Tensor::weight(
        (char *)weights->vision_patch_embed_weight,  // 权重数据指针
        meta->language_meta.dt_logits,
        shape
    );

    return vision_patch_embed_device_tensor;
}

// 创建position embedding (从meta中获取形状)
inline std::shared_ptr<Tensor> createPositionEmbedding(LlavaMeta const *meta,
    LlavaWeights const *weights) {
    // 从meta中获取参数
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;
    auto num_patches = meta->vision_meta.num_patches;

    // CLIP ViT通常还需要class token，所以位置编码长度是 num_patches + 1
    auto pos_embed_length = num_patches + 1;  // 576 + 1 = 577

    if (llava_debug_enabled_in_weighthpp()) {
        printf("[CPP createPositionEmbedding] Shape: [1, %zu, %zu]\n", pos_embed_length, vision_embed_dim);
    }

    return Tensor::weight((char *)weights->vision_position_embedding, INFINI_DTYPE_F16, {1, pos_embed_length, vision_embed_dim});
}

// 创建class token (从meta中获取形状)
inline std::shared_ptr<Tensor> getClassToken(LlavaMeta const *meta,
    LlavaWeights const *weights) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;
    
    if (llava_debug_enabled_in_weighthpp()) {
        printf("[CPP getClassToken] vision_class_token pointer: %p\n", weights->vision_class_token);
    }
    auto vision_class_token_device_tensor = 
        Tensor::weight((char *)weights->vision_class_token, 
        INFINI_DTYPE_F16, 
        {vision_embed_dim});

    // printf("[CPP getClassToken] First 10 values: \n");
    // vision_class_token_device_tensor->debug_first_n(10);
    
    return vision_class_token_device_tensor;
}

inline std::shared_ptr<Tensor> getVisionQWeight(
    LlavaMeta const *meta,
    LlavaWeights const *weights,
    size_t layer) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_q_weights[layer],
        INFINI_DTYPE_F16,
        {vision_embed_dim, vision_embed_dim}
    );
}
inline std::shared_ptr<Tensor> getVisionQBias(
    LlavaMeta const *meta,
    LlavaWeights const *weights,
    size_t layer) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_q_biases[layer],
        INFINI_DTYPE_F16,
        {vision_embed_dim}
    );
}

inline std::shared_ptr<Tensor> getVisionKWeight(
    LlavaMeta const *meta,
    LlavaWeights const *weights,
    size_t layer) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_k_weights[layer],
        INFINI_DTYPE_F16,
        {vision_embed_dim, vision_embed_dim}
    );
}
inline std::shared_ptr<Tensor> getVisionKBias(
    LlavaMeta const *meta,
    LlavaWeights const *weights,
    size_t layer) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_k_biases[layer],
        INFINI_DTYPE_F16,
        {vision_embed_dim}
    );
}
inline std::shared_ptr<Tensor> getVisionVWeight(
    LlavaMeta const *meta,
    LlavaWeights const *weights,
    size_t layer) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_v_weights[layer],
        INFINI_DTYPE_F16,
        {vision_embed_dim, vision_embed_dim}
    );
}
inline std::shared_ptr<Tensor> getVisionVBias(
    LlavaMeta const *meta,
    LlavaWeights const *weights,
    size_t layer) {
    auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_v_biases[layer],
        INFINI_DTYPE_F16,
        {vision_embed_dim}
    );
}

inline std::shared_ptr<Tensor> getVisionPreLNWeight(
    LlavaMeta const *meta,
    LlavaWeights const *weights) {

    if (llava_debug_enabled_in_weighthpp()) {
        printf("[CPP getVisionPreLNWeight] vision_pre_layernorm_weight pointer: %p\n", weights->vision_pre_layernorm_weight);
    }
    auto dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_pre_layernorm_weight,
        INFINI_DTYPE_F16,
        {dim}
    );
}

inline std::shared_ptr<Tensor> getVisionPreLNBias(
    LlavaMeta const *meta,
    LlavaWeights const *weights) {

    auto dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_pre_layernorm_bias,
        INFINI_DTYPE_F16,
        {dim}
    );
}

inline std::shared_ptr<Tensor> getVisionPostLNWeight(
    LlavaMeta const *meta,
    LlavaWeights const *weights) {

    auto dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_post_layernorm_weight,
        INFINI_DTYPE_F16,
        {dim}
    );
}

inline std::shared_ptr<Tensor> getVisionPostLNBias(
    LlavaMeta const *meta,
    LlavaWeights const *weights) {

    auto dim = meta->vision_meta.vision_embed_dim;

    return Tensor::weight(
        (char *)weights->vision_post_layernorm_bias,
        INFINI_DTYPE_F16,
        {dim}
    );
}

inline std::shared_ptr<Tensor> getVisionInLayerPreNormWeight(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    return Tensor::weight((char *)weights->vision_in_layer_pre_norm_weights[layer],
                          INFINI_DTYPE_F16, {dim});
}

inline std::shared_ptr<Tensor> getVisionInLayerPreNormBias(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    return Tensor::weight((char *)weights->vision_in_layer_pre_norm_biases[layer],
                          INFINI_DTYPE_F16, {dim});
}



inline std::shared_ptr<Tensor> getVisionProjWeight(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    return Tensor::weight((char *)weights->vision_proj_weight[layer],
                          INFINI_DTYPE_F16, {dim, dim});
}

inline std::shared_ptr<Tensor> getVisionProjBias(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    return Tensor::weight((char *)weights->vision_proj_bias[layer],
                          INFINI_DTYPE_F16, {dim});
}


inline std::shared_ptr<Tensor> getVisionInLayerPostNormWeight(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    // printf("[CPP vision_in_layer_post_norm_weight] layer: %zu, pointer: %p\n", layer, weights->vision_in_layer_post_norm_weight[layer]);
    return Tensor::weight((char *)weights->vision_in_layer_post_norm_weight[layer],
                          INFINI_DTYPE_F16, {dim});
}

inline std::shared_ptr<Tensor> getVisionInLayerPostNormBias(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    // printf("[CPP vision_post_norm_bias] layer: %zu, pointer: %p\n", layer, weights->vision_post_norm_bias[layer]);
    return Tensor::weight((char *)weights->vision_post_norm_bias[layer],
                          INFINI_DTYPE_F16, {dim});
}


inline std::shared_ptr<Tensor> getVisionMLPFC1Weight(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    auto mlp = meta->vision_meta.vision_intermediate_size;
    return Tensor::weight((char *)weights->vision_mlp_fc1_weight[layer],
                          INFINI_DTYPE_F16, {mlp, dim});
}

inline std::shared_ptr<Tensor> getVisionMLPFC1Bias(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto mlp = meta->vision_meta.vision_intermediate_size;
    return Tensor::weight((char *)weights->vision_mlp_fc1_bias[layer],
                          INFINI_DTYPE_F16, {mlp});
}


inline std::shared_ptr<Tensor> getVisionMLPFC2Weight(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    auto mlp = meta->vision_meta.vision_intermediate_size;
    return Tensor::weight((char *)weights->vision_mlp_fc2_weight[layer],
                          INFINI_DTYPE_F16, {dim, mlp});
}

inline std::shared_ptr<Tensor> getVisionMLPFC2Bias(
    LlavaMeta const *meta, LlavaWeights const *weights, size_t layer) {
    auto dim = meta->vision_meta.vision_embed_dim;
    return Tensor::weight((char *)weights->vision_mlp_fc2_bias[layer],
                          INFINI_DTYPE_F16, {dim});
}

// MultiModal Projector (two-layer MLP)
inline std::shared_ptr<Tensor> getProjectorWeight1(
    LlavaMeta const *meta, LlavaWeights const *weights) {
    auto vision_dim = meta->projector_meta.vision_embed_dim;
    auto hidden_dim = meta->projector_meta.projector_hidden_size;
    return Tensor::weight((char *)weights->projector_weight_1,
                          INFINI_DTYPE_F16, {hidden_dim, vision_dim});
}

inline std::shared_ptr<Tensor> getProjectorBias1(
    LlavaMeta const *meta, LlavaWeights const *weights) {
    auto hidden_dim = meta->projector_meta.projector_hidden_size;
    return Tensor::weight((char *)weights->projector_bias_1,
                          INFINI_DTYPE_F16, {hidden_dim});
}

inline std::shared_ptr<Tensor> getProjectorWeight2(
    LlavaMeta const *meta, LlavaWeights const *weights) {
    auto text_dim = meta->projector_meta.text_embed_dim;
    auto hidden_dim = meta->projector_meta.projector_hidden_size;
    return Tensor::weight((char *)weights->projector_weight_2,
                          INFINI_DTYPE_F16, {text_dim, hidden_dim});
}

inline std::shared_ptr<Tensor> getProjectorBias2(
    LlavaMeta const *meta, LlavaWeights const *weights) {
    auto text_dim = meta->projector_meta.text_embed_dim;
    return Tensor::weight((char *)weights->projector_bias_2,
                          INFINI_DTYPE_F16, {text_dim});
}


// inline std::shared_ptr<Tensor> createClassEmbedding(LlavaMeta const *meta) {
//     auto vision_embed_dim = meta->vision_meta.vision_embed_dim;

//     printf("[CPP createClassEmbedding] Shape: [1, %zu]\n", vision_embed_dim);

//     std::vector<float> class_embedding_data(vision_embed_dim, 0.0f);
//     return Tensor::weight(class_embedding_data.data(), INFINI_DTYPE_F16, {1, vision_embed_dim});
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
