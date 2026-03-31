#include "kv_quant.hpp"
#include "infinicore/ops/per_tensor_dequant_i8.hpp"
#include "infinicore/ops/per_tensor_quant_i8.hpp"

namespace infinilm {

void KVQuantUtils::quantize(
    infinicore::Tensor &k,
    infinicore::Tensor &v,
    infinicore::quantization::KVQuantAlgo algo,
    const infinicore::Tensor &k_scale,
    const infinicore::Tensor &v_scale) {

    if (algo == infinicore::quantization::KVQuantAlgo::NONE) {
        return;
    }

    auto device = k->device();
    auto dtype = k->dtype();
    auto zero_point = infinicore::Tensor::zeros({1}, dtype, device);

    k = infinicore::op::per_tensor_quant_i8(k, k_scale, zero_point, true);
    v = infinicore::op::per_tensor_quant_i8(v, v_scale, zero_point, true);
}

void KVQuantUtils::dequantize(
    infinicore::Tensor &k,
    infinicore::Tensor &v,
    infinicore::quantization::KVQuantAlgo algo,
    const infinicore::Tensor &k_scale,
    const infinicore::Tensor &v_scale,
    const infinicore::Tensor &reference) {

    if (algo == infinicore::quantization::KVQuantAlgo::NONE) {
        return; // 无需反量化
    }

    auto zero_point = infinicore::Tensor::zeros({1}, reference->dtype(), reference->device());

    auto k_dequant = infinicore::Tensor::strided_empty(
        k->shape(), k->strides(), reference->dtype(), reference->device());
    auto v_dequant = infinicore::Tensor::strided_empty(
        v->shape(), v->strides(), reference->dtype(), reference->device());

    infinicore::op::per_tensor_dequant_i8_(k_dequant, k, k_scale, zero_point);
    infinicore::op::per_tensor_dequant_i8_(v_dequant, v, v_scale, zero_point);

    k = std::move(k_dequant);
    v = std::move(v_dequant);
}

} // namespace infinilm
