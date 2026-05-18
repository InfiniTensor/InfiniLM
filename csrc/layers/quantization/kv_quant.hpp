#pragma once

#include "infinicore/quantization.hpp"
#include "infinicore/tensor.hpp"
#include <utility>

namespace infinilm {

class KVQuantUtils {
public:
    /**
     * @brief Quantize K/V (before writing to cache) - modifies k and v in-place
     * @param k Original K tensor
     * @param v Original V tensor
     * @param algo Quantization algorithm
     * @param k_scale Scale for K
     * @param v_scale Scale for V
     */
    static void quantize(
        infinicore::Tensor &k,
        infinicore::Tensor &v,
        infinicore::quantization::KVQuantAlgo algo,
        const infinicore::Tensor &k_scale,
        const infinicore::Tensor &v_scale);

    /**
     * @brief Dequantize K/V (after reading from cache) - modifies k and v in-place
     * @param k Quantized K tensor
     * @param v Quantized V tensor
     * @param algo Quantization algorithm
     * @param k_scale Scale for K
     * @param v_scale Scale for V
     * @param reference Reference tensor (used to obtain dtype/device)
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
