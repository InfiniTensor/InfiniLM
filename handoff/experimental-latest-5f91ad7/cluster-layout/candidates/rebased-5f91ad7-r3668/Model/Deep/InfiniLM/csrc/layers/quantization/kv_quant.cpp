#include "kv_quant.hpp"
#include "../../execution/submit.hpp"
#include "runtime/runtime.hpp"
#include <base/per_tensor_quant_i8.h>
#include <stdexcept>

namespace infinilm {
namespace {
void validate_kv(infinilm::runtime::Runtime &runtime, const infinilm::runtime::Tensor &k,
                 const infinilm::runtime::Tensor &v, const infinilm::runtime::Tensor &k_scale,
                 const infinilm::runtime::Tensor &v_scale) {
    runtime.assertOwnerThread();
    for (const auto &tensor : {k, v, k_scale, v_scale}) {
        if (!tensor || tensor->device() != runtime.device()) {
            throw std::invalid_argument("KV tensors and scales must belong to Runtime device");
        }
    }
}
} // namespace

void KVQuantUtils::quantize(
    infinilm::runtime::Runtime &runtime,
    infinilm::runtime::Tensor &k,
    infinilm::runtime::Tensor &v,
    infinilm::quantization::KVQuantAlgo algo,
    const infinilm::runtime::Tensor &k_scale,
    const infinilm::runtime::Tensor &v_scale) {

    if (algo == infinilm::quantization::KVQuantAlgo::NONE) {
        return;
    }

    if (algo != infinilm::quantization::KVQuantAlgo::INT8) {
        throw std::invalid_argument("unsupported KV quantization algorithm");
    }
    validate_kv(runtime, k, v, k_scale, v_scale);
    auto k_quant = infinilm::runtime::Tensor::strided_empty(
        runtime, k->shape(), k->strides(), infinilm::runtime::DataType::kInt8);
    execution::submit<infini::ops::PerTensorQuantI8>(
        runtime, execution::RecordingPolicy::AllowHostReplay,
        infini::ops::PerTensorQuantI8::DefaultConfig(runtime.device().type()),
        k, k_quant, k_scale, std::optional<infinilm::runtime::Tensor>{}, true);
    INFINILM_TRACE_OP(PerTensorQuantI8, k);
    auto v_quant = infinilm::runtime::Tensor::strided_empty(
        runtime, v->shape(), v->strides(), infinilm::runtime::DataType::kInt8);
    execution::submit<infini::ops::PerTensorQuantI8>(
        runtime, execution::RecordingPolicy::AllowHostReplay,
        infini::ops::PerTensorQuantI8::DefaultConfig(runtime.device().type()),
        v, v_quant, v_scale, std::optional<infinilm::runtime::Tensor>{}, true);
    INFINILM_TRACE_OP(PerTensorQuantI8, v);
    k = std::move(k_quant);
    v = std::move(v_quant);
}

void KVQuantUtils::dequantize(
    infinilm::runtime::Runtime &runtime,
    infinilm::runtime::Tensor &k,
    infinilm::runtime::Tensor &v,
    infinilm::quantization::KVQuantAlgo algo,
    const infinilm::runtime::Tensor &k_scale,
    const infinilm::runtime::Tensor &v_scale,
    const infinilm::runtime::Tensor &reference) {

    if (algo == infinilm::quantization::KVQuantAlgo::NONE) {
        return; // 无需反量化
    }

    if (algo != infinilm::quantization::KVQuantAlgo::INT8) {
        throw std::invalid_argument("unsupported KV quantization algorithm");
    }
    validate_kv(runtime, k, v, k_scale, v_scale);
    if (!reference || reference->device() != runtime.device()) {
        throw std::invalid_argument("KV dequantization reference must belong to Runtime device");
    }
    auto k_dequant = infinilm::runtime::Tensor::strided_empty(
        runtime, k->shape(), k->strides(), reference->dtype());
    auto v_dequant = infinilm::runtime::Tensor::strided_empty(
        runtime, v->shape(), v->strides(), reference->dtype());

    execution::submit<infini::ops::PerTensorDequantI8>(
        runtime, execution::RecordingPolicy::AllowHostReplay,
        infini::ops::PerTensorDequantI8::DefaultConfig(runtime.device().type()),
        k_dequant, k, k_scale, std::optional<infinilm::runtime::Tensor>{});
    INFINILM_TRACE_OP(PerTensorDequantI8, k_dequant);
    execution::submit<infini::ops::PerTensorDequantI8>(
        runtime, execution::RecordingPolicy::AllowHostReplay,
        infini::ops::PerTensorDequantI8::DefaultConfig(runtime.device().type()),
        v_dequant, v, v_scale, std::optional<infinilm::runtime::Tensor>{});
    INFINILM_TRACE_OP(PerTensorDequantI8, v_dequant);

    k = std::move(k_dequant);
    v = std::move(v_dequant);
}

} // namespace infinilm
