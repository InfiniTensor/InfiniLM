#include "infinicore.h"
#include "qwen3_vl.hpp"

#include <cassert>
#include <cmath>

inline std::shared_ptr<Tensor> getSinTable(size_t dctx, size_t dh, float theta, infiniDtype_t dtype) {
    auto half_dh = dh / 4; // 2D MRoPE: table_dim = dhead / 4
    auto unit = dsize(dtype);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            if (dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "Sin table unsupported dtype" << std::endl;
                std::abort();
            }
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, dtype, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(size_t dctx, size_t dh, float theta, infiniDtype_t dtype) {
    auto half_dh = dh / 4; // 2D MRoPE: table_dim = dhead / 4
    auto unit = dsize(dtype);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            if (dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "Cos table unsupported dtype" << std::endl;
                std::abort();
            }
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, dtype, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getSinTable_llm(size_t dctx, size_t dh, float theta, infiniDtype_t dtype) {
    auto half_dh = dh / 2; // 3dmrope sin/cos 和普通rope一样
    auto unit = dsize(dtype);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            if (dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "Sin table unsupported dtype" << std::endl;
                std::abort();
            }
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, dtype, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable_llm(size_t dctx, size_t dh, float theta, infiniDtype_t dtype) {
    auto half_dh = dh / 2; // 3dmrope sin/cos 和普通rope一样
    auto unit = dsize(dtype);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            if (dtype == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (dtype == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (dtype == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "Cos table unsupported dtype" << std::endl;
                std::abort();
            }
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, dtype, shape);
    std::free(table);
    return tensor;
}

namespace {

inline void print_info(const Qwen3VLMeta &meta) {

    printf("\nQwen3VLMeta: \n");
    // common
    printf(" dt_logits : %d\n", meta.dt_logits);
    printf(" nlayer : %ld\n", meta.nlayer);
    printf(" d : %ld\n", meta.d);
    printf(" dctx : %ld\n", meta.dctx);
    printf(" dvoc : %ld\n", meta.dvoc);
    printf(" epsilon : %f\n", meta.epsilon);
    printf(" end_token : %d\n", meta.end_token);

    // llm
    printf(" nh : %ld\n", meta.nh);
    printf(" nkvh : %ld\n", meta.nkvh);
    printf(" dh : %ld\n", meta.dh);
    printf(" theta : %f\n", meta.theta);

    // vision encoder
    printf(" vision_hidden_size : %ld\n", meta.vision_hidden_size);
    printf(" vision_layers : %ld\n", meta.vision_layers);
    printf(" vision_heads : %ld\n", meta.vision_heads);
    printf(" patch_size : %ld\n", meta.patch_size);
    printf(" img_size : %ld\n", meta.img_size);
    printf(" image_token_id : %d\n", meta.image_token_id);
    printf(" video_token_id : %d\n", meta.video_token_id);

    printf("\n");
}
}; // namespace

// Qwen3VLMeta:
//  dt_logits : 19
//  nlayer : 28
//  d : 2048
//  dctx : 1024
//  dvoc : 151936
//  epsilon : 0.000001
//  end_token : 151645
//  nh : 16
//  nkvh : 8
//  dh : 128
//  theta : 5000000.000000
//  vision_hidden_size : 1024
//  vision_layers : 24
//  vision_heads : 16
//  patch_size : 16
//  img_size : 768
//  image_token_id : 151655
//  video_token_id : 151656

//    "out_hidden_size": 2048,
//    "intermediate_size": 3072,

Qwen3VLWeights::Qwen3VLWeights(
    const Qwen3VLMeta *meta,
    infiniDevice_t device,
    const std::vector<int> &dev_ids) : infinicore::weights::Loader(device, dev_ids) {
    auto ndev = dev_ids.size();
    _device_weights.resize(ndev);
    infiniDtype_t dt_logits = meta->dt_logits;
    infiniDtype_t dt_norm_w = meta->dt_norm_w;
    size_t nlayer = meta->nlayer;
    size_t d = meta->d;
    size_t nh = meta->nh / ndev;
    size_t nkvh = meta->nkvh / ndev;
    size_t dh = meta->dh;
    size_t di = meta->di / ndev;
    size_t dctx = meta->dctx;
    size_t dvoc = meta->dvoc;

    print_info(*meta);

    // Vision encoder parameters
    size_t vision_hidden_size = meta->vision_hidden_size;
    size_t vision_layers = meta->vision_layers;
    size_t vision_heads = meta->vision_heads;
    size_t patch_size = meta->patch_size;

    for (size_t i = 0; i < ndev; i++) {
        RUN_INFINI(infinirtSetDevice(device, dev_ids[i]));

        auto weight = std::make_shared<Qwen3VLDeviceWeight>();
        _device_weights[i] = weight;

        auto w_in_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d});
        this->register_weight("model.language_model.embed_tokens.weight", w_in_embd, i);
        weight->w_in_embd = w_in_embd;

        auto w_out_norm = Tensor::weight(nullptr, dt_norm_w, {d});
        this->register_weight("model.language_model.norm.weight", w_out_norm, i);
        weight->w_out_norm = w_out_norm;

        auto w_out_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d})->permute({1, 0});
        this->register_weight("model.lm_head.weight", w_out_embd, i);
        weight->w_out_embd = w_out_embd;

        weight->sin_table = getSinTable_llm(dctx, dh, meta->theta, dt_logits);
        weight->cos_table = getCosTable_llm(dctx, dh, meta->theta, dt_logits);

        // 视觉 mRoPE 表（2D）：按 vision head 维度构建
        size_t dh_v = vision_heads > 0 ? (vision_hidden_size / vision_heads) : vision_hidden_size;
        weight->sin_table_v = getSinTable(dctx, dh_v, meta->theta, dt_logits);
        weight->cos_table_v = getCosTable(dctx, dh_v, meta->theta, dt_logits);

#define REGISTER_LAYER_WEIGHT_1D(W_NAME, W_VAR, W_DIM, W_DTYPE, W_DIST_TYPE)                     \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM});                                      \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

#define REGISTER_LAYER_WEIGHT_2D(W_NAME, W_VAR, W_DIM_1, W_DIM_2, W_DTYPE, W_DIST_TYPE)          \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM_1, W_DIM_2});                           \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

#define REGISTER_LAYER_WEIGHT_5D(W_NAME, W_VAR, W_DIM_1, W_DIM_2, W_DIM_3, W_DIM_4, W_DIM_5, W_DTYPE, W_DIST_TYPE) \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM_1, W_DIM_2, W_DIM_3, W_DIM_4, W_DIM_5});                  \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE);                   \
    weight->W_VAR.push_back(W_VAR);

// merger 权重：norm(1024) -> fc1(4096,4096) -> fc2(4096,2048)
#define REGISTER_MERGER()                                                                                                                                 \
    REGISTER_LAYER_WEIGHT_1D("model.visual.merger.norm.bias", b_v_merger_ln_q, vision_hidden_size, dt_norm_w, FULL);                                      \
    REGISTER_LAYER_WEIGHT_1D("model.visual.merger.norm.weight", w_v_merger_ln_q, vision_hidden_size, dt_norm_w, FULL);                                    \
    REGISTER_LAYER_WEIGHT_1D("model.visual.merger.linear_fc1.bias", b_v_merger_mlp_0, 4 * vision_hidden_size, dt_logits, FULL);                           \
    REGISTER_LAYER_WEIGHT_2D("model.visual.merger.linear_fc1.weight", w_v_merger_mlp_0, 4 * vision_hidden_size, 4 * vision_hidden_size, dt_logits, FULL); \
    REGISTER_LAYER_WEIGHT_1D("model.visual.merger.linear_fc2.bias", b_v_merger_mlp_2, d, dt_logits, FULL);                                                \
    REGISTER_LAYER_WEIGHT_2D("model.visual.merger.linear_fc2.weight", w_v_merger_mlp_2, d, 4 * vision_hidden_size, dt_logits, FULL);

// merger_list 权重（实际是 deepstack_merger_list）
#define REGISTER_MERGER_LIST(IDX)                                                                                                                                                                \
    REGISTER_LAYER_WEIGHT_1D("model.visual.deepstack_merger_list." #IDX ".norm.bias", b_v_merger_list_##IDX##_ln_q, 4 * vision_hidden_size, dt_norm_w, FULL);                                    \
    REGISTER_LAYER_WEIGHT_1D("model.visual.deepstack_merger_list." #IDX ".norm.weight", w_v_merger_list_##IDX##_ln_q, 4 * vision_hidden_size, dt_norm_w, FULL);                                  \
    REGISTER_LAYER_WEIGHT_1D("model.visual.deepstack_merger_list." #IDX ".linear_fc1.bias", b_v_merger_list_##IDX##_mlp_0, 4 * vision_hidden_size, dt_logits, COLUMN);                           \
    REGISTER_LAYER_WEIGHT_2D("model.visual.deepstack_merger_list." #IDX ".linear_fc1.weight", w_v_merger_list_##IDX##_mlp_0, 4 * vision_hidden_size, 4 * vision_hidden_size, dt_logits, COLUMN); \
    REGISTER_LAYER_WEIGHT_1D("model.visual.deepstack_merger_list." #IDX ".linear_fc2.bias", b_v_merger_list_##IDX##_mlp_2, d, dt_logits, COLUMN);                                                \
    REGISTER_LAYER_WEIGHT_2D("model.visual.deepstack_merger_list." #IDX ".linear_fc2.weight", w_v_merger_list_##IDX##_mlp_2, d, 4 * vision_hidden_size, dt_logits, COLUMN);

        // patch embed和pos embed权重
        REGISTER_LAYER_WEIGHT_1D("model.visual.patch_embed.proj.bias", b_v_patch_embed_proj, vision_hidden_size, dt_logits, FULL);
        REGISTER_LAYER_WEIGHT_5D("model.visual.patch_embed.proj.weight", w_v_patch_embed_proj, vision_hidden_size, 3, 2, patch_size, patch_size, dt_logits, FULL);
        REGISTER_LAYER_WEIGHT_2D("model.visual.pos_embed.weight", w_v_pos_embed, 2304, vision_hidden_size, dt_logits, FULL);

        // merger 和 deepstack_merger_list 权重
        for (size_t merger_layer = 0; merger_layer < 4; merger_layer++) {
            if (merger_layer == 0) {
                REGISTER_MERGER();
            } else {
                size_t merge_idx = merger_layer - 1;
                if (merge_idx == 0) {
                    REGISTER_MERGER_LIST(0);
                } else if (merge_idx == 1) {
                    REGISTER_MERGER_LIST(1);
                } else {
                    REGISTER_MERGER_LIST(2);
                }
            }
        }

        for (size_t layer = 0; layer < nlayer; layer++) {

            // vision encoder blocks
            if (layer < vision_layers) {
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".norm1.bias", b_v_norm1, vision_hidden_size, dt_norm_w, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".norm1.weight", w_v_norm1, vision_hidden_size, dt_norm_w, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".attn.proj.bias", b_v_attn_proj, vision_hidden_size, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_2D("model.visual.blocks." + std::to_string(layer) + ".attn.proj.weight", w_v_attn_proj, vision_hidden_size, vision_hidden_size, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".attn.qkv.bias", b_v_attn_qkv, 3 * vision_hidden_size, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_2D("model.visual.blocks." + std::to_string(layer) + ".attn.qkv.weight", w_v_attn_qkv, 3 * vision_hidden_size, vision_hidden_size, dt_logits, FULL);

                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".norm2.bias", b_v_norm2, vision_hidden_size, dt_norm_w, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".norm2.weight", w_v_norm2, vision_hidden_size, dt_norm_w, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".mlp.linear_fc1.bias", b_v_mlp_fc1, 4 * vision_hidden_size, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_2D("model.visual.blocks." + std::to_string(layer) + ".mlp.linear_fc1.weight", w_v_mlp_fc1, 4 * vision_hidden_size, vision_hidden_size, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.visual.blocks." + std::to_string(layer) + ".mlp.linear_fc2.bias", b_v_mlp_fc2, vision_hidden_size, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_2D("model.visual.blocks." + std::to_string(layer) + ".mlp.linear_fc2.weight", w_v_mlp_fc2, vision_hidden_size, 4 * vision_hidden_size, dt_logits, FULL);
            }

            // language model layers
            REGISTER_LAYER_WEIGHT_1D("model.language_model.layers." + std::to_string(layer) + ".input_layernorm.weight", w_attn_norm, d, dt_norm_w, FULL);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight", w_attn_q, d, nh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight", w_attn_k, d, nkvh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_1D("model.language_model.layers." + std::to_string(layer) + ".self_attn.q_norm.weight", w_q_norm, nh * dh, dt_norm_w, FULL);
            REGISTER_LAYER_WEIGHT_1D("model.language_model.layers." + std::to_string(layer) + ".self_attn.k_norm.weight", w_k_norm, nkvh * dh, dt_norm_w, FULL);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight", w_attn_v, d, nkvh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight", w_attn_out, nh * dh, d, dt_logits, ROW);

            REGISTER_LAYER_WEIGHT_1D("model.language_model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight", w_ffn_norm, d, dt_norm_w, FULL);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".mlp.gate_proj.weight", w_ffn_gate, d, di, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".mlp.up_proj.weight", w_ffn_up, d, di, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.language_model.layers." + std::to_string(layer) + ".mlp.down_proj.weight", w_ffn_down, di, d, dt_logits, ROW);
        }
    }

#undef REGISTER_LAYER_WEIGHT_1D
#undef REGISTER_LAYER_WEIGHT_2D
#undef REGISTER_MERGER
#undef REGISTER_MERGER_LIST
}

__C struct ModelWeights *
createQwen3VLWeights(const Qwen3VLMeta *meta,
                     infiniDevice_t device,
                     int ndev,
                     const int *dev_ids) {
    Qwen3VLWeights *weights = new Qwen3VLWeights(meta, device, std::vector<int>(dev_ids, dev_ids + ndev));
    return (struct ModelWeights *)weights;
}
