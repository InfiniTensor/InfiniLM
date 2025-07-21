#include "core/inf_assert.h"
#include "infiniop/ops/moe_dispatch.h"
#include "moe_dispatch_info.h"
#include <cuda_runtime.h>
#include <iostream>

namespace infiniop {

struct infiniopMoEDispatchDescriptor : public infiniopDescriptor {
    MoEDispatchInfo info;
    infiniopMoEDispatchDescriptor(infiniopHandle_t handle,
                                int num_experts,
                                infiniopTensorDescriptor_t input_desc,
                                infiniopTensorDescriptor_t indices_desc,
                                infiniopTensorDescriptor_t permuted_output_desc,
                                infiniopTensorDescriptor_t aux_info_desc)
        : infiniopDescriptor(handle, INFOP_OP_MOE_DISPATCH),
          info(num_experts, input_desc, indices_desc, permuted_output_desc, aux_info_desc) {}
};

} // namespace infiniop

void infiniopCreateMoEDispatchDescriptor(
    infiniopHandle_t handle,
    infiniopMoEDispatchDescriptor_t *desc,
    int num_experts,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t permuted_output_desc,
    infiniopTensorDescriptor_t aux_info_desc) {
    *desc = new infiniop::infiniopMoEDispatchDescriptor(handle, num_experts, input_desc, indices_desc, permuted_output_desc, aux_info_desc);
}

void infiniopDestroyMoEDispatchDescriptor(infiniopMoEDispatchDescriptor_t desc) {
    delete desc;
}

void infiniopMoEDispatch(infiniopMoEDispatchDescriptor_t desc,
                       void *permuted_output,
                       void *aux_info,
                       const void *input,
                       const void *indices,
                       infiniopStream_t stream) {
    desc->info.op(permuted_output, aux_info, input, indices, stream);
} 