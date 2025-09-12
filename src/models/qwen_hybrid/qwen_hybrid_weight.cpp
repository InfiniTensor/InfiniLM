#include "qwen_hybrid.hpp"

#include <cmath>

inline std::shared_ptr<Tensor> getSinTable(size_t dctx, size_t dh, float theta, infiniDtype_t dtype) {
    auto half_dh = dh / 2;
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
    auto half_dh = dh / 2;
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

inline void print_info(const QwenHybridMeta &meta) {

    printf("\nQwenHybridMeta: \n");
    // common
    printf(" dt_logits : %d\n", meta.dtype);
    printf(" nlayer : %ld\n", meta.nlayer);
    printf(" d : %ld\n", meta.d);
    printf(" dctx : %ld\n", meta.dctx);
    printf(" dvoc : %ld\n", meta.dvoc);
    printf(" epsilon : %f\n", meta.epsilon);
    printf(" end_token : %d\n", meta.end_token);

    // mha
    printf(" nh : %ld\n", meta.nh);
    printf(" nkvh : %ld\n", meta.nkvh);
    printf(" dh : %ld\n", meta.dh);
    printf(" theta : %f\n", meta.theta);

    // moe
    printf(" nexperts : %ld\n", meta.nexperts);
    printf(" kexperts : %ld\n", meta.kexperts);
    printf(" shared_di : %ld\n", meta.shared_di);
    printf(" moe_di : %ld\n", meta.moe_di);
    printf(" norm_topk_prob : %d\n", meta.norm_topk_prob);

    printf("\n");
}
}; // namespace

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
    size_t nkvh = meta->nkvh >= ndev ? meta->nkvh / ndev : 1;
    size_t dh = meta->dh;
    // size_t di = meta->shared_di / ndev;
    size_t l_n_k_head = meta->l_n_k_head / ndev;
    size_t l_n_v_head = meta->l_n_v_head / ndev;
    size_t l_k_dim = meta->l_k_dim;
    size_t l_v_dim = meta->l_v_dim;
    size_t l_k_size = l_n_k_head * l_k_dim;
    size_t l_v_size = l_n_v_head * l_v_dim;
    size_t l_conv_kernel_dim = meta->l_conv_kernel_dim;
    size_t projection_size_qkvz = l_n_k_head * l_k_dim * 2 + l_n_v_head * l_v_dim * 2;
    size_t projection_size_ba = l_n_v_head * 2;
    size_t dctx = meta->dctx;
    size_t dvoc = meta->dvoc;

    // print_info(*meta);

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

        weight->sin_table = getSinTable(dctx, dh / 4, meta->theta, dt_logits);
        weight->cos_table = getCosTable(dctx, dh / 4, meta->theta, dt_logits);

        // 先resize每个layer的空间
        weight->w_router_expert_ffn_gate.resize(nlayer);
        weight->w_router_expert_ffn_up.resize(nlayer);
        weight->w_router_expert_ffn_down.resize(nlayer);

        for (size_t layer = 0; layer < nlayer; layer++) {

#define REGISTER_LAYER_WEIGHT_1D(W_NAME, W_VAR, W_DIM, W_DTYPE, W_DIST_TYPE)                     \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM});                                      \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

#define REGISTER_LAYER_WEIGHT_2D(W_NAME, W_VAR, W_DIM_1, W_DIM_2, W_DTYPE, W_DIST_TYPE)          \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM_2, W_DIM_1})->permute({1, 0});          \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

#define REGISTER_LAYER_WEIGHT_3D(W_NAME, W_VAR, W_DIM_1, W_DIM_2, W_DIM_3, W_DTYPE, W_DIST_TYPE) \
    auto W_VAR = Tensor::weight(nullptr, W_DTYPE, {W_DIM_1, W_DIM_2, W_DIM_3});                  \
    this->register_weight(W_NAME, W_VAR, i, infinicore::weights::DistributionType::W_DIST_TYPE); \
    weight->W_VAR.push_back(W_VAR);

            REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".input_layernorm.weight", w_attn_norm, d, dt_norm_w, FULL);

            if (layer % 4 < 3) {
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".linear_attn.dt_bias", b_la_dt, l_n_v_head, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".linear_attn.A_log", alpha_la_g, l_n_v_head, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_3D("model.layers." + std::to_string(layer) + ".linear_attn.conv1d.weight", w_la_conv, 2 * l_k_size + l_v_size, 1, l_conv_kernel_dim, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".linear_attn.in_proj_qkvz.weight", w_la_qkvz, d, projection_size_qkvz, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".linear_attn.in_proj_ba.weight", w_la_ba, d, projection_size_ba, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".linear_attn.norm.weight", w_la_norm, l_v_dim, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".linear_attn.out_proj.weight", w_la_out, l_n_v_head * l_v_dim, d, dt_logits, COLUMN);
                weight->w_attn_q.push_back(nullptr);
                weight->w_attn_k.push_back(nullptr);
                weight->w_attn_v.push_back(nullptr);
                weight->w_attn_out.push_back(nullptr);
                weight->b_attn_q.push_back(nullptr);
                weight->b_attn_k.push_back(nullptr);
                weight->b_attn_v.push_back(nullptr);
                weight->w_attn_q_norm.push_back(nullptr);
                weight->w_attn_k_norm.push_back(nullptr);
            } else {
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.q_proj.weight", w_attn_q, d, nh * dh * 2, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.k_proj.weight", w_attn_k, d, nkvh * dh, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.v_proj.weight", w_attn_v, d, nkvh * dh, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_2D("model.layers." + std::to_string(layer) + ".self_attn.o_proj.weight", w_attn_out, nh * dh, d, dt_logits, COLUMN);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.q_proj.bias", b_attn_q, nh * dh * 2, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.k_proj.bias", b_attn_k, nkvh * dh, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.v_proj.bias", b_attn_v, nkvh * dh, dt_logits, ROW);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.q_norm.weight", w_attn_q_norm, dh, dt_logits, FULL);
                REGISTER_LAYER_WEIGHT_1D("model.layers." + std::to_string(layer) + ".self_attn.k_norm.weight", w_attn_k_norm, dh, dt_logits, FULL);
                weight->b_la_dt.push_back(nullptr);
                weight->alpha_la_g.push_back(nullptr);
                weight->w_la_conv.push_back(nullptr);
                weight->w_la_qkvz.push_back(nullptr);
                weight->w_la_ba.push_back(nullptr);
                weight->w_la_norm.push_back(nullptr);
                weight->w_la_out.push_back(nullptr);
            }

            // MoE
            std::string name = "model.layers." + std::to_string(layer) + ".post_attention_layernorm.weight";
            REGISTER_LAYER_WEIGHT_1D(name, w_ffn_norm, d, dt_norm_w, FULL);
            {
                // gate
                name = "model.layers." + std::to_string(layer) + ".mlp.shared_expert_gate.weight";
                REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_gate, d, 1, dt_logits, FULL);

                name = "model.layers." + std::to_string(layer) + ".mlp.gate.weight";
                REGISTER_LAYER_WEIGHT_2D(name, w_router_expert_gate, d, meta->nexperts, dt_logits, FULL);
            }

            {
                // 共享专家
                size_t shared_di = meta->shared_di / ndev;

                name = "model.layers." + std::to_string(layer) + ".mlp.shared_expert.gate_proj.weight";
                REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_ffn_gate, d, shared_di, dt_logits, ROW);

                name = "model.layers." + std::to_string(layer) + ".mlp.shared_expert.up_proj.weight";
                REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_ffn_up, d, shared_di, dt_logits, ROW);

                name = "model.layers." + std::to_string(layer) + ".mlp.shared_expert.down_proj.weight";
                REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_ffn_down, shared_di, d, dt_logits, COLUMN);
            }

            {
                // 路由专家
                size_t moe_di = meta->moe_di / ndev;
                for (size_t iexpert = 0; iexpert < meta->nexperts; ++iexpert) {

                    // REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_ffn_gate, d, moe_di, dt_logits, ROW);
                    {
                        name = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(iexpert) + ".gate_proj.weight";
                        auto gate = Tensor::weight(nullptr, dt_logits, {moe_di, d})->permute({1, 0});
                        this->register_weight(name, gate, i, infinicore::weights::DistributionType::ROW);
                        weight->w_router_expert_ffn_gate[layer].push_back(gate);
                    }

                    // REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_ffn_up, d, moe_di, dt_logits, ROW);
                    {
                        name = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(iexpert) + ".up_proj.weight";
                        auto up = Tensor::weight(nullptr, dt_logits, {moe_di, d})->permute({1, 0});
                        this->register_weight(name, up, i, infinicore::weights::DistributionType::ROW);
                        weight->w_router_expert_ffn_up[layer].push_back(up);
                    }

                    // REGISTER_LAYER_WEIGHT_2D(name, w_shared_expert_ffn_down, moe_di, d, dt_logits, COLUMN);
                    {
                        name = "model.layers." + std::to_string(layer) + ".mlp.experts." + std::to_string(iexpert) + ".down_proj.weight";
                        auto down = Tensor::weight(nullptr, dt_logits, {d, moe_di})->permute({1, 0});
                        this->register_weight(name, down, i, infinicore::weights::DistributionType::COLUMN);
                        weight->w_router_expert_ffn_down[layer].push_back(down);
                    }
                }
            }
        }
    }

#undef REGISTER_LAYER_WEIGHT_1D
#undef REGISTER_LAYER_WEIGHT_2D
#undef REGISTER_LAYER_WEIGHT_3D
}

__C struct ModelWeights *
createQwenHybridWeights(const QwenHybridMeta *meta,
                        infiniDevice_t device,
                        int ndev,
                        const int *dev_ids) {
    QwenHybridWeights *weights = new QwenHybridWeights(meta, device, std::vector<int>(dev_ids, dev_ids + ndev));
    return (struct ModelWeights *)weights;
}
