#ifndef LLADAMOE_WEIGHT_HPP
#define LLADAMOE_WEIGHT_HPP


#include <cmath>
void debugPrint(LLaDAMeta const *meta) {
    if (!meta) {
        std::cout << "LLaDAMeta pointer = NULL" << std::endl;
        return;
    }

    std::cout << "===== LLaDAMeta DEBUG =====" << std::endl;
    std::cout << "meta pointer   = " << meta << std::endl;

    std::cout << "dt_logits      = " << (int)meta->dt_logits << std::endl;
    std::cout << "nlayer         = " << meta->nlayer << std::endl;
    std::cout << "d              = " << meta->d << std::endl;
    std::cout << "nh             = " << meta->nh << std::endl;
    std::cout << "nkvh           = " << meta->nkvh << std::endl;
    std::cout << "dh             = " << meta->dh << std::endl;

    std::cout << "di_dense       = " << meta->di_dense << std::endl;
    std::cout << "di_expert      = " << meta->di_expert << std::endl;

    std::cout << "dctx           = " << meta->dctx << std::endl;
    std::cout << "dvoc           = " << meta->dvoc << std::endl;

    std::cout << "epsilon        = " << meta->epsilon << std::endl;
    std::cout << "theta          = " << meta->theta << std::endl;

    std::cout << "end_token      = " << meta->end_token << std::endl;
    std::cout << "num_experts    = " << meta->num_experts << std::endl;

    std::cout << "===========================" << std::endl;
}

inline std::shared_ptr<Tensor> getInEmbd(
    LLaDAMeta const * meta,
    LLaDAWeights const * w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    LLaDAMeta const * meta,
    LLaDAWeights const * w){
    auto shape = std::vector<size_t>({meta->d}); //TODO:
    return Tensor::weight((char *)w->output_norm, w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    LLaDAMeta const *meta,
    LLaDAWeights const *w) {
    std::cout << "Out Embd sd" << std::endl;

    auto shape = std::vector<size_t>({meta->d, meta->dvoc});
    return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getAttnNorm(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnQKV(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    // size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(w->dt_mat);
    auto shape = std::vector<size_t>({d * 3, d});
    auto t = Tensor::weight((char *)(w->attn_qkv[layer]), w->dt_mat, shape);
    // std::cout << "Load Debug" << std::endl;
    // std::cout << t->info() << std::endl;
    return t;
    // if (w->transpose_linear_weights != 0) {
    //     auto shape = std::vector<size_t>({d * 3, d});
    //     return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape)
    //         ->permute({1, 0});
    // } else {
    //     auto shape = std::vector<size_t>({d, (nh + 2 * nkvh) / ndev * dh});
    //     return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, w->dt_mat, shape);
    // }
}



inline std::shared_ptr<Tensor> getAttnQNorm(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->dh});
    // // std::cout << "QWQWW" << std::endl;
    // Tensor::weight((char *)(w->attn_q_norm[layer]), w->dt_norm, shape)->debug();
    return Tensor::weight((char *)(w->attn_q_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnKNorm(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->dh});
    return Tensor::weight((char *)(w->attn_k_norm[layer]), w->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnO(LLaDAMeta const *meta,
                                        LLaDAWeights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh) * dsize(w->dt_mat);
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({d, nh / ndev * dh});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({nh / ndev * dh, d});
        return Tensor::weight((char *)(w->attn_o[layer]) + offset, w->dt_mat, shape);
    }
}

inline std::shared_ptr<Tensor> getFFNNorm(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), w->dt_norm, shape);
}

// inline std::shared_ptr<Tensor> getFFNGateUp(
//     LLaDAMeta const *meta,
//     LLaDAWeights const *w,
//     size_t layer, size_t idev, size_t ndev) {
//     auto di = meta->di_expert; // TODO: 具体di还要区分
//     auto d = meta->d;
//     size_t offset = idev * (2 * di / ndev) * d * dsize(w->dt_mat);
//     if (w->transpose_linear_weights != 0) {
//         auto shape = std::vector<size_t>({2 * di / ndev, d});
//         return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
//                               w->dt_mat, shape)
//             ->permute({1, 0});
//     } else {
//         auto shape = std::vector<size_t>({d, 2 * di / ndev});
//         return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
//                               w->dt_mat, shape);
//     }
// }
//
inline std::shared_ptr<Tensor> getExpertRouter(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer, size_t idev, size_t ndev, size_t num_experts) {
    auto shape = std::vector<size_t>({meta->d});
    auto di = meta->di_expert; // TODO: 具体di还要区分
    auto d = meta->d;
    size_t offset = 0;
    if (w->transpose_linear_weights != 0) {
        auto shape = std::vector<size_t>({num_experts, d});
        return Tensor::weight((char *)(w->expert_router[layer]) + offset,
                              w->dt_mat, shape)
            ->permute({1, 0});
    } else {
        auto shape = std::vector<size_t>({num_experts, d});
        return Tensor::weight((char *)(w->expert_router[layer]) + offset,
                              w->dt_mat, shape)->permute({1, 0});
    }
}

inline std::shared_ptr<Tensor> getExpertGate(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer, size_t idev, size_t ndev){
    auto di = meta->di_expert; 
    auto d = meta->d;
    size_t offset = 0;
    auto shape = std::vector<size_t>({meta->num_experts, d, di});
    return Tensor::weight((char *)(w->expert_gate[layer]) + offset,
                              meta->dt_logits, shape);
    
}

inline std::shared_ptr<Tensor> getExpertUp(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di_expert; // TODO: 具体di还要区分
    auto d = meta->d;
    size_t offset = 0;
    auto shape = std::vector<size_t>({meta->num_experts, d, di});
    return Tensor::weight((char *)(w->expert_up[layer]) + offset,
                              meta->dt_logits, shape);
}


inline std::shared_ptr<Tensor> getExpertDown(
    LLaDAMeta const *meta,
    LLaDAWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di_expert; // TODO: 具体di还要区分
    auto d = meta->d;
    size_t offset = 0;
    auto shape = std::vector<size_t>({meta->num_experts, di, d});
    return Tensor::weight((char *)(w->expert_down[layer]) + offset,
                              w->dt_mat, shape);
}




inline std::shared_ptr<Tensor> getSinTable(LLaDAMeta const *meta) {
    std::cout << "Get Sin Table" << std::endl;
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_sin);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _sin;
            } else {
                std::cout << "unsupported data type" << std::endl;
                std::cout << meta->dt_logits << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    tensor->debug("/home/featurize/work/My_InfiniLM/layer_0_weights/sin_table.bin");
    std::cout << "Sin Table Initing  Over" << std::endl;
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(LLaDAMeta const *meta) {
    auto half_dh = meta->dh / 2;
    auto unit = dsize(meta->dt_logits);
    void *table = std::malloc(meta->dctx * half_dh * unit);

    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            if (meta->dt_logits == INFINI_DTYPE_F16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_f16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_BF16) {
                ((uint16_t *)table)[i * half_dh + j] = f32_to_bf16(_cos);
            } else if (meta->dt_logits == INFINI_DTYPE_F32) {
                ((float *)table)[i * half_dh + j] = _cos;
            } else {
                std::cout << "unsupported data type" << std::endl;
                exit(1);
            }
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, half_dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
        tensor->debug("/home/featurize/work/My_InfiniLM/layer_0_weights/cos_table.bin");
    std::free(table);
    return tensor;
}

#endif
