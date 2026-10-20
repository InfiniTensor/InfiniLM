#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <optional>

#include "native/cuda/nvidia/ops/moe_align_block_size/kernel.cuh"
#include "native/cuda/nvidia/ops/moe_align_block_size/kernel.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`MoeAlignBlockSize` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`MoeAlignBlockSize` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`MoeAlignBlockSize` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

}  // namespace

Operator<MoeAlignBlockSize, Device::Type::kNvidia, 0>::Operator(
    const Tensor topk_ids, const std::optional<Tensor> expert_map,
    const int64_t num_experts, const int64_t block_size,
    Tensor sorted_token_ids, Tensor experts_ids, Tensor num_tokens_post_pad)
    : MoeAlignBlockSize{topk_ids,           expert_map,      num_experts,
                        block_size,         sorted_token_ids, experts_ids,
                        num_tokens_post_pad},
      device_index_{topk_ids.device().index()} {}

void Operator<MoeAlignBlockSize, Device::Type::kNvidia, 0>::Run(
    const Tensor topk_ids, const std::optional<Tensor> maybe_expert_map,
    const int64_t num_experts, const int64_t block_size,
    Tensor sorted_token_ids, Tensor experts_ids,
    Tensor num_tokens_post_pad) const {
  DeviceGuard device_guard{device_index_};

  constexpr int32_t kThreads = 256;
  const auto shared_memory_size =
      static_cast<size_t>(num_experts_) * sizeof(int32_t);
  moe_align_block_size_detail::MoeAlignBlockSizeKernel<<<
      1, kThreads, shared_memory_size, static_cast<cudaStream_t>(stream_)>>>(
      reinterpret_cast<const int32_t*>(topk_ids.data()),
      maybe_expert_map
          ? reinterpret_cast<const int32_t*>(maybe_expert_map->data())
          : nullptr,
      reinterpret_cast<int32_t*>(sorted_token_ids.data()),
      reinterpret_cast<int32_t*>(experts_ids.data()),
      reinterpret_cast<int32_t*>(num_tokens_post_pad.data()), numel_,
      static_cast<int32_t>(num_experts_), static_cast<int32_t>(block_size_),
      sorted_token_ids_size_, experts_ids_size_,
      maybe_expert_map ? static_cast<int32_t>(maybe_expert_map->numel()) : 0);

  const auto status = cudaGetLastError();
  assert(status == cudaSuccess &&
         "`MoeAlignBlockSize` CUDA kernel launch failed");
}

}  // namespace infini::ops
