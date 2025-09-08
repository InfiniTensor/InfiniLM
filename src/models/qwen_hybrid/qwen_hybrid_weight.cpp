#include "qwen_hybrid.hpp"

#include <cmath>

inline std::shared_ptr<Tensor> getSinTable(size_t dctx, size_t dh, float theta) {
    auto half_dh = dh / 2;
    auto unit = dsize(INFINI_DTYPE_F16);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, INFINI_DTYPE_F16, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(size_t dctx, size_t dh, float theta) {
    auto half_dh = dh / 2;
    auto unit = dsize(INFINI_DTYPE_F16);
    void *table = std::malloc(dctx * half_dh * unit);

    for (size_t i = 0; i < dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(theta, static_cast<float>(j) / half_dh));

            ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
        }
    }
    auto shape = std::vector<size_t>({dctx, half_dh});
    auto tensor = Tensor::weight(table, INFINI_DTYPE_F16, shape);
    std::free(table);
    return tensor;
}

QwenHybridWeights::QwenHybridWeights(
    const QwenHybridMeta *meta,
    infiniDevice_t device,
    const std::vector<int> &dev_ids) : infinicore::weights::Loader(device, dev_ids) {
    auto ndev = dev_ids.size();
    _device_weights.resize(ndev);
    infiniDtype_t dt_logits = meta->dtype;
    infiniDtype_t dt_norm_w = meta->dtype;
    size_t nlayer = meta->nlayer;
    size_t d = meta->d;
    size_t nh = meta->nh / ndev;
    size_t nkvh = meta->nkvh / ndev;
    size_t dh = meta->dh;
    size_t di = meta->moe_di / ndev;
    size_t dctx = meta->dctx;
    size_t dvoc = meta->dvoc;

    for (size_t i = 0; i < ndev; i++) {
        RUN_INFINI(infinirtSetDevice(device, dev_ids[i]));

        auto weight = std::make_shared<QwenHybridDeviceWeight>();
        _device_weights[i] = weight;

        auto w_in_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d});
        this->register_weight("model.embed_tokens.weight", w_in_embd, i);
        weight->w_in_embd = w_in_embd;

        auto w_out_norm = Tensor::weight(nullptr, dt_norm_w, {d});
        this->register_weight("model.norm.weight", w_out_norm, i);
        weight->w_out_norm = w_out_norm;

        auto w_out_embd = Tensor::weight(nullptr, dt_logits, {dvoc, d})->permute({1, 0});
        this->register_weight("lm_head.weight", w_out_embd, i);
        weight->w_out_embd = w_out_embd;

        weight->sin_table = getSinTable(dctx, dh, meta->theta);
        weight->cos_table = getCosTable(dctx, dh, meta->theta);

        for (size_t layer = 0; layer < nlayer; layer++) {

#define REGISTER_LAYER_WEIGHT_1D(W_NAME, W_VAR, W_DIM, W_DTYPE, W_DIST_TYPE)                     \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM});                                      \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

#define REGISTER_LAYER_WEIGHT_2D(W_NAME, W_VAR, W_DIM_1, W_DIM_2, W_DTYPE, W_DIST_TYPE)          \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM_1, W_DIM_2});                           \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

            REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".input_layernorm.weight", w_attn_norm, d, dt_norm_w, FULL);

            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.q_proj", w_attn_q, d, nh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.k_proj", w_attn_k, d, nkvh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.v_proj", w_attn_v, d, nkvh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.q_proj.bias", b_attn_q, nh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.k_proj.bias", b_attn_k, nkvh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.v_proj.bias", b_attn_v, nkvh * dh, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.o_proj", w_attn_out, nh * dh, d, dt_logits, ROW);

            REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight", w_ffn_norm, d, dt_norm_w, FULL);
            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".mlp.gate_proj", w_ffn_gate, d, di, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".mlp.up_proj", w_ffn_up, d, di, dt_logits, COLUMN);
            REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".mlp.down_proj", w_ffn_down, di, d, dt_logits, ROW);
        }
    }

#undef RIGISTER_LAYER_WEIGHT
#undef REGISTER_LAYER_QUANT_WEIGHT
}

__C struct ModelWeights *
createQwenHybridWeights(const QwenHybridMeta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids) {
    QwenHybridWeights *weights = new QwenHybridWeights(meta, device, std::vector<int>(dev_ids, dev_ids + ndev));
    return (struct ModelWeights *)weights;
}
