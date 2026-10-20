#ifndef INFINI_OPS_CUDA_MOE_SUM_KERNEL_CUH_
#define INFINI_OPS_CUDA_MOE_SUM_KERNEL_CUH_

#include <cstdint>

#include "native/cuda/caster.cuh"

namespace infini::ops {

template <Device::Type kDevice, typename Data, typename Index>
__global__ void MoeSumKernel(
    const Data* __restrict__ input, const Index* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_map, Data* __restrict__ output,
    int64_t topk, int64_t hidden_size, int64_t input_token_stride,
    int64_t input_slot_stride, int64_t input_hidden_stride,
    int64_t topk_ids_token_stride, int64_t topk_ids_slot_stride,
    int64_t expert_map_size, int64_t expert_map_stride) {
  const int64_t token = static_cast<int64_t>(blockIdx.x);

  for (int64_t hidden = static_cast<int64_t>(threadIdx.x); hidden < hidden_size;
       hidden += static_cast<int64_t>(blockDim.x)) {
    float sum = 0.0f;

    for (int64_t slot = 0; slot < topk; ++slot) {
      if (topk_ids != nullptr) {
        const auto expert_id =
            static_cast<int64_t>(topk_ids[token * topk_ids_token_stride +
                                          slot * topk_ids_slot_stride]);
        if (expert_id < 0) {
          continue;
        }
        if (expert_map != nullptr &&
            (expert_id >= expert_map_size ||
             expert_map[expert_id * expert_map_stride] < 0)) {
          continue;
        }
      }

      const auto offset = token * input_token_stride +
                          slot * input_slot_stride +
                          hidden * input_hidden_stride;
      sum += Caster<kDevice>::template Cast<float>(input[offset]);
    }

    output[token * hidden_size + hidden] =
        Caster<kDevice>::template Cast<Data>(sum);
  }
}

}  // namespace infini::ops

#endif  // INFINI_OPS_CUDA_MOE_SUM_KERNEL_CUH_
