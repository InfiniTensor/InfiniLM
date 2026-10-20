#ifndef INFINI_OPS_CUDA_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <cassert>
#include <cstdint>
#include <optional>

#include "base/fused_add_rms_norm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/fused_add_rms_norm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaFusedAddRmsNorm : public FusedAddRmsNorm {
 public:
  using FusedAddRmsNorm::FusedAddRmsNorm;

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight, float) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 AllCudaBlockSizes>(
        {static_cast<int64_t>(input.dtype()), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto weight_data = weight.has_value()
                                 ? reinterpret_cast<const T*>(weight->data())
                                 : nullptr;
          FusedAddRmsNormKernel<kBlockSize, Backend::kDeviceType, float, T>
              <<<static_cast<uint32_t>(num_tokens_), kBlockSize, 0,
                 cuda_stream>>>(reinterpret_cast<T*>(input.data()),
                                input_strides_[input_strides_.size() - 2],
                                reinterpret_cast<T*>(residual.data()),
                                residual_strides_[residual_strides_.size() - 2],
                                weight_data, dim_, epsilon_);
        },
        "CudaFusedAddRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
