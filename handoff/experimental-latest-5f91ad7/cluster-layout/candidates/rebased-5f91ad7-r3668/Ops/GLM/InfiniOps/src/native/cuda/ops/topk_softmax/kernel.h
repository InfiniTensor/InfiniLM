#ifndef INFINI_OPS_CUDA_TOPK_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CUDA_TOPK_SOFTMAX_KERNEL_H_

#include <cstdint>
#include <optional>

#include "base/topk_softmax.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/ops/topk_softmax/kernel.cuh"

namespace infini::ops {

template <typename Backend>
class CudaTopkSoftmax : public TopkSoftmax {
 public:
  using TopkSoftmax::TopkSoftmax;

  void operator()(const Tensor gating_output, std::optional<Tensor> bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  Tensor topk_weights, Tensor topk_indices,
                  Tensor token_expert_indices) const override {
    ValidateCallMetadata(gating_output, bias, is_padding, renormalize,
                         topk_weights, topk_indices, token_expert_indices);
    if (num_tokens_ == 0) {
      return;
    }

    constexpr unsigned int kBlockSize = 256;
    using InputTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;
    using IndexTypes =
        List<DataType::kInt32, DataType::kUInt32, DataType::kInt64>;
    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    DispatchFunc<InputTypes, IndexTypes>(
        {static_cast<int64_t>(input_dtype_),
         static_cast<int64_t>(index_dtype_)},
        [&](auto list_tag) {
          using Input = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using Index = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;

          topk_softmax_detail::TopkSoftmaxKernel<
              kBlockSize, Backend::kDeviceType, Input, Index>
              <<<static_cast<unsigned int>(num_tokens_), kBlockSize, 0,
                 stream>>>(
                  reinterpret_cast<const Input*>(gating_output.data()),
                  bias ? reinterpret_cast<const float*>(bias->data()) : nullptr,
                  is_padding
                      ? reinterpret_cast<const uint8_t*>(is_padding->data())
                      : nullptr,
                  reinterpret_cast<float*>(topk_weights.data()),
                  reinterpret_cast<Index*>(topk_indices.data()),
                  reinterpret_cast<int32_t*>(token_expert_indices.data()),
                  static_cast<int32_t>(num_tokens_),
                  static_cast<int32_t>(num_experts_),
                  static_cast<int32_t>(topk_), renormalize_, 0.0f);
        },
        "CudaTopkSoftmax::operator()");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_TOPK_SOFTMAX_KERNEL_H_
