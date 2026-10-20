#pragma once

#include "quantization_scheme.hpp"
#include "runtime/tensor.hpp"
#include <utility>

namespace infinilm::runtime { class Runtime; }

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
        infinilm::runtime::Runtime &runtime,
        infinilm::runtime::Tensor &k, infinilm::runtime::Tensor &v,
        infinilm::quantization::KVQuantAlgo algo,
        const infinilm::runtime::Tensor &k_scale, const infinilm::runtime::Tensor &v_scale);

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
        infinilm::runtime::Runtime &runtime,
        infinilm::runtime::Tensor &k, infinilm::runtime::Tensor &v,
        infinilm::quantization::KVQuantAlgo algo,
        const infinilm::runtime::Tensor &k_scale, const infinilm::runtime::Tensor &v_scale,
        const infinilm::runtime::Tensor &reference);
};

} // namespace infinilm
