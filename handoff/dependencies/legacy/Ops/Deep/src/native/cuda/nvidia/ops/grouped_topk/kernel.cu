#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/nvidia/ops/grouped_topk/kernel.cuh"
#include "native/cuda/nvidia/ops/grouped_topk/kernel.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`GroupedTopk` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`GroupedTopk` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`GroupedTopk` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

}  // namespace

void Operator<GroupedTopk, Device::Type::kNvidia, 0>::operator()(
    const Tensor scores, const Tensor bias, const int64_t num_expert_group,
    const int64_t topk_group, const int64_t topk, const bool renormalize,
    const double routed_scaling_factor, const int64_t scoring_func,
    Tensor topk_values, Tensor topk_indices) const {
  ValidateCallMetadata(scores, bias, num_expert_group, topk_group, topk,
                       renormalize, routed_scaling_factor, scoring_func,
                       topk_values, topk_indices);

  if (num_tokens_ == 0) {
    return;
  }

  DeviceGuard device_guard{device_index_};
  const auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
  using DataTypes = ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>;

  DispatchFunc<DataTypes, DataTypes>(
      {static_cast<int64_t>(scores_dtype_), static_cast<int64_t>(bias_dtype_)},
      [&](auto list_tag) {
        using Score = TypeMapType<Device::Type::kNvidia, ListGet<0>(list_tag)>;
        using Bias = TypeMapType<Device::Type::kNvidia, ListGet<1>(list_tag)>;

        if (scoring_func_ == 0) {
          grouped_topk_detail::GroupedTopkKernel<Score, Bias, 0>
              <<<static_cast<unsigned int>(num_tokens_),
                 grouped_topk_detail::kBlockSize, 0, stream>>>(
                  reinterpret_cast<const Score*>(scores.data()),
                  reinterpret_cast<const Bias*>(bias.data()),
                  reinterpret_cast<float*>(topk_values.data()),
                  reinterpret_cast<int32_t*>(topk_indices.data()),
                  static_cast<int32_t>(num_experts_),
                  static_cast<int32_t>(num_expert_group_),
                  static_cast<int32_t>(topk_group_),
                  static_cast<int32_t>(topk_), renormalize_,
                  routed_scaling_factor_);
        } else {
          grouped_topk_detail::GroupedTopkKernel<Score, Bias, 1>
              <<<static_cast<unsigned int>(num_tokens_),
                 grouped_topk_detail::kBlockSize, 0, stream>>>(
                  reinterpret_cast<const Score*>(scores.data()),
                  reinterpret_cast<const Bias*>(bias.data()),
                  reinterpret_cast<float*>(topk_values.data()),
                  reinterpret_cast<int32_t*>(topk_indices.data()),
                  static_cast<int32_t>(num_experts_),
                  static_cast<int32_t>(num_expert_group_),
                  static_cast<int32_t>(topk_group_),
                  static_cast<int32_t>(topk_), renormalize_,
                  routed_scaling_factor_);
        }
      },
      "Operator<GroupedTopk, Device::Type::kNvidia>::operator()");

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess && "`GroupedTopk` CUDA kernel launch failed");
}

}  // namespace infini::ops
