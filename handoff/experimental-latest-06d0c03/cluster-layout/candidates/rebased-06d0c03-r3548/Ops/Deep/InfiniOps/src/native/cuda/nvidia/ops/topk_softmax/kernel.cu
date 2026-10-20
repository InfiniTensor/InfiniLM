#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <optional>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/caster.cuh"
#include "native/cuda/ops/topk_softmax/kernel.cuh"
#include "native/cuda/nvidia/ops/topk_softmax/kernel.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`TopkSoftmax` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`TopkSoftmax` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`TopkSoftmax` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

}  // namespace

void Operator<TopkSoftmax, Device::Type::kNvidia, 0>::operator()(
    const Tensor gating_output, std::optional<Tensor> bias,
    std::optional<Tensor> is_padding, const bool renormalize,
    Tensor topk_weights, Tensor topk_indices,
    Tensor token_expert_indices) const {
  ValidateCallMetadata(gating_output, bias, is_padding, renormalize,
                       topk_weights, topk_indices, token_expert_indices);

  if (num_tokens_ == 0) {
    return;
  }

  DeviceGuard device_guard{device_index_};
  constexpr unsigned int kBlockSize = 256;
  using InputTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;
  using IndexTypes =
      List<DataType::kInt32, DataType::kUInt32, DataType::kInt64>;
  const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);

  DispatchFunc<InputTypes, IndexTypes>(
      {static_cast<int64_t>(input_dtype_), static_cast<int64_t>(index_dtype_)},
      [&](auto list_tag) {
        using Input = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
        using Index = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;

        topk_softmax_detail::TopkSoftmaxKernel<
            kBlockSize, Device::Type::kNvidia, Input, Index>
            <<<static_cast<unsigned int>(num_tokens_), kBlockSize, 0, stream>>>(
                reinterpret_cast<const Input*>(gating_output.data()),
                bias ? reinterpret_cast<const float*>(bias->data()) : nullptr,
                is_padding
                    ? reinterpret_cast<const uint8_t*>(is_padding->data())
                    : nullptr,
                reinterpret_cast<float*>(topk_weights.data()),
                reinterpret_cast<Index*>(topk_indices.data()),
                reinterpret_cast<int32_t*>(token_expert_indices.data()),
                static_cast<int32_t>(num_tokens_),
                static_cast<int32_t>(num_experts_), static_cast<int32_t>(topk_),
                renormalize_, softcap_, min_sum_);
      },
      "Operator<TopkSoftmax, Device::Type::kNvidia>::operator()");

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess && "`TopkSoftmax` CUDA kernel launch failed");
}

}  // namespace infini::ops
