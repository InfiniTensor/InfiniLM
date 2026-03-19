#pragma once

#include "infinicore/quantization.hpp"
#include "infinicore/tensor.hpp"
#include <utility>

namespace infinilm {

class KVQuantUtils {
public:
    /**
     * @brief 量化 K/V（写入缓存前）- 原地修改 k 和 v
     * @param k 原始 K 张量
     * @param v 原始 V 张量
     * @param algo 量化算法
     * @param k_scale K 的 scale
     * @param v_scale V 的 scale
     */
    static void quantize(
        infinicore::Tensor &k,
        infinicore::Tensor &v,
        infinicore::quantization::KVQuantAlgo algo,
        const infinicore::Tensor &k_scale,
        const infinicore::Tensor &v_scale);

    /**
     * @brief 反量化 K/V（读取缓存后）- 原地修改 k 和 v
     * @param k 量化后的 K 张量
     * @param v 量化后的 V 张量
     * @param algo 量化算法
     * @param k_scale K 的 scale
     * @param v_scale V 的 scale
     * @param reference 参考张量（用于获取 dtype/device）
     */
    static void dequantize(
        infinicore::Tensor &k,
        infinicore::Tensor &v,
        infinicore::quantization::KVQuantAlgo algo,
        const infinicore::Tensor &k_scale,
        const infinicore::Tensor &v_scale,
        const infinicore::Tensor &reference);
};

} // namespace infinilm
