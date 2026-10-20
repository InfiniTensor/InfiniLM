#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <optional>

#include "native/cuda/nvidia/ops/get_cutlass_moe_mm_data/kernel.cuh"
#include "native/cuda/nvidia/ops/get_cutlass_moe_mm_data/kernel.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`GetCutlassMoeMmData` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`GetCutlassMoeMmData` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`GetCutlassMoeMmData` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

}  // namespace

void Operator<GetCutlassMoeMmData, Device::Type::kNvidia, 0>::operator()(
    const Tensor topk_ids, const int64_t num_experts, const int64_t n,
    const int64_t k, const bool is_gated, Tensor expert_offsets,
    Tensor problem_sizes1, Tensor problem_sizes2, Tensor input_permutation,
    Tensor output_permutation, std::optional<Tensor> blockscale_offsets) const {
  ValidateCallMetadata(topk_ids, num_experts, n, k, is_gated, expert_offsets,
                       problem_sizes1, problem_sizes2, input_permutation,
                       output_permutation, blockscale_offsets);

  DeviceGuard device_guard{device_index_};
  constexpr int32_t kThreads = 512;
  auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);

  get_cutlass_moe_mm_data_detail::
      GetCutlassMoeMmDataKernel<<<1, kThreads, 0, stream>>>(
          reinterpret_cast<const int32_t*>(topk_ids.data()),
          reinterpret_cast<int32_t*>(expert_offsets.data()),
          reinterpret_cast<int32_t*>(problem_sizes1.data()),
          reinterpret_cast<int32_t*>(problem_sizes2.data()),
          reinterpret_cast<int32_t*>(input_permutation.data()),
          reinterpret_cast<int32_t*>(output_permutation.data()),
          blockscale_offsets
              ? reinterpret_cast<int32_t*>(blockscale_offsets->data())
              : nullptr,
          static_cast<int64_t>(numel_), static_cast<int32_t>(topk_),
          static_cast<int32_t>(num_experts_), static_cast<int32_t>(n_),
          static_cast<int32_t>(k_), is_gated_);

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess &&
         "`GetCutlassMoeMmData` CUDA kernel launch failed");
}

}  // namespace infini::ops
