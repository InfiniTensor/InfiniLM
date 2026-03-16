#ifndef QWEN3VL_WEIGHTS_H
#define QWEN3VL_WEIGHTS_H

#include <infiniccl.h>
#include <infiniop.h>
#include <infinirt.h>

#include <stddef.h>
#include <stdint.h>

struct Qwen3vlWeights;

// Function pointer signatures
typedef void (*qwen3vl_load_global_fn)(Qwen3vlWeights *, void *cpu_ptr);
typedef void (*qwen3vl_load_layer_fn)(Qwen3vlWeights *, void *cpu_ptr, size_t layer_id);
// Struct containing all weight loading functions
typedef struct {
    // Global
    qwen3vl_load_global_fn load_input_embd;
    qwen3vl_load_global_fn load_output_norm;
    qwen3vl_load_global_fn load_output_embd;

    // Attention
    qwen3vl_load_layer_fn load_attn_norm;
    qwen3vl_load_layer_fn load_attn_q_norm;
    qwen3vl_load_layer_fn load_attn_k_norm;
    qwen3vl_load_layer_fn load_attn_qkv_proj;
    qwen3vl_load_layer_fn load_attn_o_proj;
    
    // MLP
    qwen3vl_load_layer_fn load_mlp_norm;
    qwen3vl_load_layer_fn load_mlp_gate_up;
    qwen3vl_load_layer_fn load_mlp_down;

} Qwen3vlLangWeightLoader;

typedef struct {
    // Patch_embed
    qwen3vl_load_global_fn load_patch_embed_weight;
    qwen3vl_load_global_fn load_patch_embed_bias;
    qwen3vl_load_global_fn load_pos_embed_weight;

    // blocks attn
    qwen3vl_load_layer_fn load_attn_proj_weight;
    qwen3vl_load_layer_fn load_attn_proj_bias;
    qwen3vl_load_layer_fn load_attn_qkv_weight;
    qwen3vl_load_layer_fn load_attn_qkv_bias;

    //block mlp
    qwen3vl_load_layer_fn load_mlp_linear_fc1_weight;
    qwen3vl_load_layer_fn load_mlp_linear_fc1_bias;
    qwen3vl_load_layer_fn load_mlp_linear_fc2_weight;
    qwen3vl_load_layer_fn load_mlp_linear_fc2_bias;

    //block norm
    qwen3vl_load_layer_fn  load_norm1_weight;
    qwen3vl_load_layer_fn  load_norm1_bias;
    qwen3vl_load_layer_fn  load_norm2_weight;
    qwen3vl_load_layer_fn  load_norm2_bias;

    //deepstack_merger
    qwen3vl_load_layer_fn load_deepstack_merger_linear_fc1_weight;
    qwen3vl_load_layer_fn load_deepstack_merger_linear_fc1_bias;
    qwen3vl_load_layer_fn load_deepstack_merger_linear_fc2_weight;
    qwen3vl_load_layer_fn load_deepstack_merger_linear_fc2_bias;
    qwen3vl_load_layer_fn load_deepstack_merger_norm_weight;
    qwen3vl_load_layer_fn load_deepstack_merger_norm_bias;

    //merger
    qwen3vl_load_global_fn load_merger_linear_fc1_weight;
    qwen3vl_load_global_fn load_merger_linear_fc1_bias;
    qwen3vl_load_global_fn load_merger_linear_fc2_weight;
    qwen3vl_load_global_fn load_merger_linear_fc2_bias;
    qwen3vl_load_global_fn load_merger_norm_weight;
    qwen3vl_load_global_fn load_merger_norm_bias;

} Qwen3vlVisWeightLoader;

typedef struct { 
    Qwen3vlLangWeightLoader lang_loader;
    Qwen3vlVisWeightLoader vis_loader;
} Qwen3vlWeightLoader;

struct Qwen3vlModel;

typedef struct {
    size_t bos_token_id;
    size_t eos_token_id;
    size_t head_dim;
    size_t hidden_size;
    float initializer_range;
    size_t intermediate_size;
    size_t max_tokens;
    size_t num_attention_heads;
    size_t num_hidden_layers;
    size_t num_key_value_heads;
    float rms_norm_eps;
    size_t mrope_section[3];
    size_t rope_theta;
    size_t vocab_size;
} Qwen3vlTextMeta;

typedef struct {
    size_t depth;
    size_t deepstack_visual_indexes[3];
    size_t hidden_size;
    size_t in_channels;
    float initializer_range;
    size_t intermediate_size;
    size_t num_heads;
    size_t num_position_embeddings;
    size_t out_hidden_size;
    size_t patch_size;
    size_t spatial_merge_size;
    size_t temporal_patch_size;
} Qwen3vlVisMeta;

typedef struct {
    infiniDtype_t dtype; //INFINI_DTYPE_BF16

    Qwen3vlTextMeta text_meta;
    Qwen3vlVisMeta vis_meta;

    size_t image_token_id;
    size_t video_token_id;
    size_t vision_end_token_id;
    size_t vision_start_token_id;
} Qwen3vlMeta;

//////////////////// APIs ///////////////////////
/// @brief 创建模型
/// @param device 协处理器种类
/// @param ndev 协处理器数量
/// @param dev_ids 协处理器编号，长度为 ndev
__C __export struct Qwen3vlModel *
createQwen3vlModel(const Qwen3vlMeta *,
                      const Qwen3vlWeights *);

__C Qwen3vlWeights *
createQwen3vlWeights(const Qwen3vlMeta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids,
                        bool transpose_weight);

__C __export Qwen3vlWeightLoader *
createQwen3vlWeightLoader();

/// @brief 销毁模型
__C __export void destroyQwen3vlModel(struct Qwen3vlModel *);

__C __export struct Qwen3vlCache *
createQwen3vlCache(const struct Qwen3vlModel *);

__C __export void
dropQwen3vlCache(const struct Qwen3vlModel *,
                    struct Qwen3vlCache *);

/// @brief 批次推理一轮，并采样出新的 token
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param temperature 采样温度（0. 表示贪心采样）
/// @param topk 采样 topk（1 表示贪心采样）
/// @param topp 采样 topp
/// @param output 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
inferBatchQwen3vl(struct Qwen3vlModel *,
                    const uint32_t *tokens, uint32_t ntok,
                    void *pixel_values, uint32_t total_patches,
                    uint32_t *image_grid_thw, uint32_t num_images,
                    void *pixel_values_videos, uint32_t total_patches_videos,
                    uint32_t *video_grid_thw, uint32_t num_videos,
                    uint32_t patch_features,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct Qwen3vlCache **caches,
                    const float *temperature, const uint32_t *topk, const float *topp,
                    uint32_t *output);

/// @brief 批次推理一轮，输出 output embedding 后的 logits
/// @param tokens 输入 token 地址
/// @param ntok 输入 token 数量
/// @param nreq 请求数量
/// @param req_lens 每个请求的 token 数量
/// @param req_pos 每个请求的起始位置
/// @param kv_caches 每个请求的 KV Cache
/// @param logits 输出 token 数组，每个请求一个输出，长度至少为nreq
__C __export void
forwardBatchQwen3vl(struct Qwen3vlModel *,
                    const uint32_t *tokens, uint32_t ntok,
                    void *pixel_values, uint32_t total_patches,
                    uint32_t *image_grid_thw, uint32_t num_images,
                    void *pixel_values_videos, uint32_t total_patches_videos,
                    uint32_t *video_grid_thw, uint32_t num_videos,
                    uint32_t patch_features,
                    const uint32_t *req_lens, uint32_t nreq, const uint32_t *req_pos,
                    struct Qwen3vlCache **caches,
                    void *logits);

#endif // QWEN3VL_WEIGHTS_H
