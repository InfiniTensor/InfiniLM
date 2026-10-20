// SPDX-License-Identifier: Apache-2.0
// Adapted from vLLM at commit ffc4f08c8ee130d4ea6347c1bf31ffd4f8af28ab:
// csrc/libtorch_stable/moe/moe_wna16.cu

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cassert>
#include <cstdint>
#include <optional>

#include "native/cuda/nvidia/ops/moe_wna16_gemm/kernel.cuh"
#include "native/cuda/nvidia/ops/moe_wna16_gemm/kernel.h"

namespace infini::ops {
namespace {

class DeviceGuard {
 public:
  explicit DeviceGuard(int device_index) {
    auto status = cudaGetDevice(&previous_device_);
    assert(status == cudaSuccess &&
           "`MoeWna16Gemm` failed to query the current CUDA device");

    if (previous_device_ != device_index) {
      status = cudaSetDevice(device_index);
      assert(status == cudaSuccess &&
             "`MoeWna16Gemm` failed to select the input CUDA device");
      restore_ = true;
    }
  }

  ~DeviceGuard() {
    if (restore_) {
      const auto status = cudaSetDevice(previous_device_);
      assert(status == cudaSuccess &&
             "`MoeWna16Gemm` failed to restore the CUDA device");
    }
  }

 private:
  int previous_device_{0};

  bool restore_{false};
};

template <typename Data, int kBit>
void Launch(const Tensor input, const Tensor b_qweight, const Tensor b_scales,
            std::optional<Tensor> b_qzeros, std::optional<Tensor> topk_weights,
            const Tensor sorted_token_ids, const Tensor expert_ids,
            const Tensor num_tokens_post_pad, Tensor output, int64_t m,
            int64_t n, int64_t k, int64_t group_size, int64_t top_k,
            int64_t block_size_m, int64_t block_size_n, int64_t block_size_k,
            cudaStream_t stream) {
  auto effective_sorted_size = sorted_token_ids.numel();
  if (m <= block_size_m) {
    const auto limit =
        static_cast<decltype(effective_sorted_size)>(m * block_size_m * top_k);
    if (effective_sorted_size > limit) {
      effective_sorted_size = limit;
    }
  }
  const auto num_token_blocks =
      (effective_sorted_size + block_size_m - 1) / block_size_m;
  const dim3 grid(
      static_cast<unsigned int>(num_token_blocks),
      static_cast<unsigned int>((n + block_size_n - 1) / block_size_n),
      static_cast<unsigned int>((k + block_size_k - 1) / block_size_k));
  const dim3 block(static_cast<unsigned int>(block_size_n));
  const auto shared_memory_size =
      static_cast<size_t>(block_size_m * block_size_k * sizeof(Data));

  const auto* qzeros =
      b_qzeros ? reinterpret_cast<const uint32_t*>(b_qzeros->data()) : nullptr;
  const auto* weights =
      topk_weights ? reinterpret_cast<const float*>(topk_weights->data())
                   : nullptr;
  const auto groups_per_block = block_size_k / group_size;

#define LAUNCH_MOE_WNA16_GEMM(GROUPS)                                      \
  moe_wna16_gemm_detail::MoeWna16GemmKernel<Data, kBit, GROUPS>            \
      <<<grid, block, shared_memory_size, stream>>>(                       \
          reinterpret_cast<const Data*>(input.data()),                     \
          reinterpret_cast<Data*>(output.data()),                          \
          reinterpret_cast<const uint32_t*>(b_qweight.data()),             \
          reinterpret_cast<const Data*>(b_scales.data()), qzeros, weights, \
          reinterpret_cast<const int32_t*>(sorted_token_ids.data()),       \
          reinterpret_cast<const int32_t*>(expert_ids.data()),             \
          reinterpret_cast<const int32_t*>(num_tokens_post_pad.data()),    \
          static_cast<uint64_t>(sorted_token_ids.numel()),                 \
          static_cast<uint16_t>(group_size), static_cast<uint16_t>(top_k), \
          static_cast<uint32_t>(m), static_cast<uint32_t>(n),              \
          static_cast<uint32_t>(k), static_cast<uint16_t>(block_size_m),   \
          static_cast<uint16_t>(block_size_n),                             \
          static_cast<uint16_t>(block_size_k), b_qzeros.has_value(),       \
          topk_weights.has_value())

  switch (groups_per_block) {
    case 1:
      LAUNCH_MOE_WNA16_GEMM(1);
      break;
    case 2:
      LAUNCH_MOE_WNA16_GEMM(2);
      break;
    case 4:
      LAUNCH_MOE_WNA16_GEMM(4);
      break;
    case 8:
      LAUNCH_MOE_WNA16_GEMM(8);
      break;
    default:
      assert(false && "`MoeWna16Gemm` received an unsupported group count");
  }

#undef LAUNCH_MOE_WNA16_GEMM
}

}  // namespace

Operator<MoeWna16Gemm, Device::Type::kNvidia, 0>::Operator(
    const Tensor input, const Tensor b_qweight, const Tensor b_scales,
    std::optional<Tensor> b_qzeros, std::optional<Tensor> topk_weights,
    const Tensor sorted_token_ids, const Tensor expert_ids,
    const Tensor num_tokens_post_pad, const int64_t top_k,
    const int64_t block_size_m, const int64_t block_size_n,
    const int64_t block_size_k, const int64_t bit, Tensor output)
    : MoeWna16Gemm{input,        b_qweight,
                   b_scales,     b_qzeros,
                   topk_weights, sorted_token_ids,
                   expert_ids,   num_tokens_post_pad,
                   top_k,        block_size_m,
                   block_size_n, block_size_k,
                   bit,          output} {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device_index_);
  assert(status == cudaSuccess &&
         "`MoeWna16Gemm` failed to query the CUDA device");
  assert(properties.major >= 7 &&
         "`MoeWna16Gemm` requires compute capability 7.0 or newer");
  assert((dtype_ != DataType::kBFloat16 || properties.major >= 8) &&
         "`MoeWna16Gemm` requires compute capability 8.0 for bfloat16");
  assert(block_size_n_ <= properties.maxThreadsPerBlock &&
         block_size_m_ * block_size_k_ * 2 <= properties.sharedMemPerBlock &&
         "`MoeWna16Gemm` block geometry exceeds CUDA device limits");
}

void Operator<MoeWna16Gemm, Device::Type::kNvidia, 0>::operator()(
    const Tensor input, const Tensor b_qweight, const Tensor b_scales,
    std::optional<Tensor> b_qzeros, std::optional<Tensor> topk_weights,
    const Tensor sorted_token_ids, const Tensor expert_ids,
    const Tensor num_tokens_post_pad, const int64_t top_k,
    const int64_t block_size_m, const int64_t block_size_n,
    const int64_t block_size_k, const int64_t bit, Tensor output) const {
  ValidateCallMetadata(input, b_qweight, b_scales, b_qzeros, topk_weights,
                       sorted_token_ids, expert_ids, num_tokens_post_pad, top_k,
                       block_size_m, block_size_n, block_size_k, bit, output);

  DeviceGuard device_guard{device_index_};
  auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : nullptr);
  const auto output_bytes =
      static_cast<size_t>(m_ * top_k_ * n_ * sizeof(uint16_t));
  auto status = cudaMemsetAsync(output.data(), 0, output_bytes, stream);
  assert(status == cudaSuccess &&
         "`MoeWna16Gemm` failed to clear the output tensor");

  if (dtype_ == DataType::kFloat16 && bit_ == 4) {
    Launch<half, 4>(input, b_qweight, b_scales, b_qzeros, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_pad, output,
                    m_, n_, k_, group_size_, top_k_, block_size_m_,
                    block_size_n_, block_size_k_, stream);
  } else if (dtype_ == DataType::kFloat16) {
    Launch<half, 8>(input, b_qweight, b_scales, b_qzeros, topk_weights,
                    sorted_token_ids, expert_ids, num_tokens_post_pad, output,
                    m_, n_, k_, group_size_, top_k_, block_size_m_,
                    block_size_n_, block_size_k_, stream);
  } else if (bit_ == 4) {
    Launch<__nv_bfloat16, 4>(
        input, b_qweight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
        expert_ids, num_tokens_post_pad, output, m_, n_, k_, group_size_,
        top_k_, block_size_m_, block_size_n_, block_size_k_, stream);
  } else {
    Launch<__nv_bfloat16, 8>(
        input, b_qweight, b_scales, b_qzeros, topk_weights, sorted_token_ids,
        expert_ids, num_tokens_post_pad, output, m_, n_, k_, group_size_,
        top_k_, block_size_m_, block_size_n_, block_size_k_, stream);
  }

  status = cudaGetLastError();
  assert(status == cudaSuccess && "`MoeWna16Gemm` CUDA kernel launch failed");
}

}  // namespace infini::ops
