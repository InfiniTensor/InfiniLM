#ifndef MODEL_MINICPMV_H
#define MODEL_MINICPMV_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stdint.h>

struct MiniCPMVModel;

typedef struct {
    // Vision encoder (SigLIP-NaViT)
    size_t patch_size;              // 14
    size_t vision_embed_dim;        // 1152
    size_t vision_num_layers;       // 27
    size_t vision_num_heads;        // 16
    size_t vision_intermediate_size;// 4304
    float vision_layer_norm_eps;    // 1e-6
    size_t vision_image_size;       // 980 (pos-embed grid: 70x70)
    size_t vision_num_positions;    // 4900
} MiniCPMVVisionMeta;

typedef struct {
    // Resampler (Perceiver-style, one cross-attn)
    size_t num_queries;         // 64
    size_t embed_dim;           // 3584
    size_t num_heads;           // 28
    size_t kv_dim;              // 1152
    float layer_norm_eps;       // 1e-6
    size_t max_patches_h;       // 70
    size_t max_patches_w;       // 70
} MiniCPMVResamplerMeta;

typedef struct {
    // Language model meta (same layout as JiugeMeta)
    infiniDtype_t dt_logits;
    size_t nlayer, d, nh, nkvh, dh, di, dctx, dvoc;
    float epsilon, theta;
    uint32_t end_token;
} MiniCPMVLanguageMeta;

typedef struct {
    MiniCPMVVisionMeta vision_meta;
    MiniCPMVResamplerMeta resampler_meta;
    MiniCPMVLanguageMeta language_meta;
} MiniCPMVMeta;

typedef struct {
    // LayerNorm
    const void *layer_norm1_weight;
    const void *layer_norm1_bias;
    const void *layer_norm2_weight;
    const void *layer_norm2_bias;

    // Self-attention
    const void *q_weight;
    const void *q_bias;
    const void *k_weight;
    const void *k_bias;
    const void *v_weight;
    const void *v_bias;
    const void *out_weight;
    const void *out_bias;

    // MLP
    const void *fc1_weight;
    const void *fc1_bias;
    const void *fc2_weight;
    const void *fc2_bias;
} MiniCPMVSiglipLayerWeights;

typedef struct {
    // SigLIP patch embedding conv2d
    const void *vpm_patch_embedding_weight;  // [vision_embed_dim, 3, patch, patch]
    const void *vpm_patch_embedding_bias;    // [vision_embed_dim]
    // SigLIP position embedding
    const void *vpm_position_embedding;      // [vision_num_positions, vision_embed_dim]
    // SigLIP encoder layers
    const MiniCPMVSiglipLayerWeights *vpm_layers;  // [vision_num_layers]
    // SigLIP final LN
    const void *vpm_post_layernorm_weight;   // [vision_embed_dim]
    const void *vpm_post_layernorm_bias;     // [vision_embed_dim]

    // Resampler
    const void *resampler_query;             // [num_queries, embed_dim]
    // NOTE: For the current CPU reference implementation, these weights are expected
    // to be pre-transposed to "in x out" layout for GEMM: [in_dim, out_dim].
    const void *resampler_kv_proj_weight;      // [kv_dim, embed_dim]
    const void *resampler_attn_in_proj_weight; // [embed_dim, 3*embed_dim]
    const void *resampler_attn_in_proj_bias;   // [3*embed_dim]
    const void *resampler_attn_out_proj_weight;  // [embed_dim, embed_dim]
    const void *resampler_attn_out_proj_bias;   // [embed_dim]
    const void *resampler_ln_q_weight;       // [embed_dim]
    const void *resampler_ln_q_bias;         // [embed_dim]
    const void *resampler_ln_kv_weight;      // [embed_dim]
    const void *resampler_ln_kv_bias;        // [embed_dim]
    const void *resampler_ln_post_weight;    // [embed_dim]
    const void *resampler_ln_post_bias;      // [embed_dim]
    const void *resampler_proj;                // [embed_dim, embed_dim]

    // Language model weights (reuse Jiuge layout)
    size_t nlayer;
    infiniDtype_t dt_norm, dt_mat;
    int transpose_linear_weights;

    const void *input_embd;
    const void *output_norm;
    const void *output_embd;
    const void *const *attn_norm;
    const void *const *attn_qkv;
    const void *const *attn_qkv_b;
    const void *const *attn_q_norm;
    const void *const *attn_k_norm;
    const void *const *attn_o;
    const void *const *ffn_norm;
    const void *const *ffn_gate_up;
    const void *const *ffn_down;
} MiniCPMVWeights;

//////////////////// APIs ///////////////////////
__C __export struct MiniCPMVModel *
createMiniCPMVModel(const MiniCPMVMeta *meta,
                    const MiniCPMVWeights *weights,
                    infiniDevice_t device,
                    int ndev,
                    const int *dev_ids);

__C __export void
destroyMiniCPMVModel(struct MiniCPMVModel *model);

/// @brief Resampler forward (CPU reference path).
/// @note This API is for step-by-step validation; it currently supports CPU only.
/// @param x Vision features, shape [seq_len, kv_dim], dtype = meta.language_meta.dt_logits
/// @param seq_len Must equal tgt_h * tgt_w (no padding supported in this API)
/// @param tgt_h Patch grid height
/// @param tgt_w Patch grid width
/// @param output Output, shape [num_queries, embed_dim], dtype = meta.language_meta.dt_logits
__C __export void
inferMiniCPMVResampler(struct MiniCPMVModel *model,
                       const void *x, size_t seq_len,
                       uint32_t tgt_h, uint32_t tgt_w,
                       void *output);

/// @brief SigLIP patch embedding + position embedding (CPU reference path).
/// @note This API is for step-by-step validation; it currently supports CPU only.
/// @param pixel_values Input packed as [1, 3, patch_size, seq_len * patch_size], where seq_len == tgt_h * tgt_w.
/// @param output Output embeddings, shape [seq_len, vision_embed_dim], dtype = meta.language_meta.dt_logits
__C __export void
inferMiniCPMVSiglipEmbeddings(struct MiniCPMVModel *model,
                             const void *pixel_values,
                             size_t seq_len,
                             uint32_t tgt_h,
                             uint32_t tgt_w,
                             void *output);

/// @brief SigLIP encoder layer0 forward (CPU reference path).
/// @note This API is for step-by-step validation; it currently supports CPU only and ignores attention masks.
/// @param hidden_states Input, shape [seq_len, vision_embed_dim]
/// @param output Output, shape [seq_len, vision_embed_dim]
__C __export void
inferMiniCPMVSiglipLayer0(struct MiniCPMVModel *model,
                          const void *hidden_states,
                          size_t seq_len,
                          void *output);

/// @brief SigLIP encoder layer forward (CPU reference path).
/// @note This API is for step-by-step validation; it currently supports CPU only and ignores attention masks.
/// @param layer_idx Which encoder layer to run.
__C __export void
inferMiniCPMVSiglipLayer(struct MiniCPMVModel *model,
                         uint32_t layer_idx,
                         const void *hidden_states,
                         size_t seq_len,
                         void *output);

/// @brief SigLIP encoder forward for the first `num_layers` layers, followed by post-layernorm (CPU reference path).
/// @note This API is for step-by-step validation; it currently supports CPU only and ignores attention masks.
__C __export void
inferMiniCPMVSiglipEncoder(struct MiniCPMVModel *model,
                           uint32_t num_layers,
                           const void *hidden_states,
                           size_t seq_len,
                           void *output);

/// @brief Vision forward: SigLIP embeddings -> SigLIP encoder -> resampler (CPU reference path).
/// @note This API is for step-by-step validation; it currently supports CPU only.
/// @param pixel_values Input packed as [1, 3, patch_size, seq_len * patch_size], where seq_len == tgt_h * tgt_w.
/// @param output Output, shape [num_queries, embed_dim], dtype = meta.language_meta.dt_logits
__C __export void
inferMiniCPMVVisionResampler(struct MiniCPMVModel *model,
                             const void *pixel_values,
                             size_t seq_len,
                             uint32_t tgt_h,
                             uint32_t tgt_w,
                             void *output);

#endif
