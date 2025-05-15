#ifndef JIUGE_WEIGHT_HPP
#define JIUGE_WEIGHT_HPP

#include "jiuge_impl.hpp"

#include <cmath>
inline std::shared_ptr<Tensor> getInEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->input_embd, meta->dt_logits, shape);
}

inline std::shared_ptr<Tensor> getOutNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)w->output_norm, meta->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getOutEmbd(
    JiugeMeta const *meta,
    JiugeWeights const *w) {
    auto shape = std::vector<size_t>({meta->dvoc, meta->d});
    return Tensor::weight((char *)w->output_embd, meta->dt_logits, shape)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> getAttnNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->attn_norm[layer]), meta->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getAttnQKV(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * d * dsize(meta->dt_mat);
    auto shape = std::vector<size_t>({(nh + 2 * nkvh) / ndev * dh, d});
    return Tensor::weight((char *)(w->attn_qkv[layer]) + offset, meta->dt_mat, shape)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> getAttnQKVBias(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto nkvh = meta->nkvh;
    auto nh = meta->nh;
    auto dh = meta->dh;
    size_t offset = idev * ((nkvh * 2 + nh) / ndev * dh) * dsize(meta->dt_mat);
    auto shape = std::vector<size_t>({1, (nh + 2 * nkvh) / ndev * dh});
    return Tensor::weight((char *)(w->attn_qkv_b[layer]) + offset, meta->dt_mat, shape);
}

inline std::shared_ptr<Tensor> getAttnO(JiugeMeta const *meta,
                                        JiugeWeights const *w, size_t layer,
                                        size_t idev, size_t ndev) {
    auto nh = meta->nh;
    auto dh = meta->dh;
    auto d = meta->d;
    size_t offset = idev * d * (nh / ndev * dh) * dsize(meta->dt_mat);
    auto shape = std::vector<size_t>({d, nh / ndev * dh});
    return Tensor::weight((char *)(w->attn_o[layer]) + offset, meta->dt_mat, shape)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> getFFNNorm(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer) {
    auto shape = std::vector<size_t>({meta->d});
    return Tensor::weight((char *)(w->ffn_norm[layer]), meta->dt_norm, shape);
}

inline std::shared_ptr<Tensor> getFFNGateUp(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * (2 * di / ndev) * d * dsize(meta->dt_mat);
    auto shape = std::vector<size_t>({2 * di / ndev, d});
    return Tensor::weight((char *)(w->ffn_gate_up[layer]) + offset,
                          meta->dt_mat, shape)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> getFFNDown(
    JiugeMeta const *meta,
    JiugeWeights const *w,
    size_t layer, size_t idev, size_t ndev) {
    auto di = meta->di;
    auto d = meta->d;
    size_t offset = idev * d * (di / ndev) * dsize(meta->dt_mat);
    auto shape = std::vector<size_t>({d, di / ndev});
    return Tensor::weight((char *)(w->ffn_down[layer]) + offset, meta->dt_mat, shape)
        ->permute({1, 0});
}

inline std::shared_ptr<Tensor> getSinTable(JiugeMeta const *meta) {
    float *table = (float *)std::malloc(meta->dctx * meta->dh * sizeof(float));
    auto half_dh = meta->dh / 2;
    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _sin = std::sin(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            table[i * meta->dh + 2 * j] = _sin;
            table[i * meta->dh + 2 * j + 1] = _sin;
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, meta->dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

inline std::shared_ptr<Tensor> getCosTable(JiugeMeta const *meta) {
    float *table = (float *)std::malloc(meta->dctx * meta->dh * sizeof(float));
    auto half_dh = meta->dh / 2;
    for (size_t i = 0; i < meta->dctx; i++) {
        for (size_t j = 0; j < half_dh; j++) {
            float _cos = std::cos(
                static_cast<float>(i) / std::pow(meta->theta, static_cast<float>(j) / half_dh));
            table[i * meta->dh + 2 * j] = _cos;
            table[i * meta->dh + 2 * j + 1] = _cos;
        }
    }
    auto shape = std::vector<size_t>({meta->dctx, meta->dh});
    auto tensor = Tensor::weight(table, meta->dt_logits, shape);
    std::free(table);
    return tensor;
}

#endif
