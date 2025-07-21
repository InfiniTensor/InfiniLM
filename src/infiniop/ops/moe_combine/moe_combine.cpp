#include "core/inf_assert.h"
#include "infiniop/ops/moe_combine.h"
#include "moe_combine_info.h"
#include <cuda_runtime.h>
#include <iostream>

namespace infiniop {

struct infiniopMoECombineDescriptor : public infiniopDescriptor {
    MoECombineInfo info;
    infiniopMoECombineDescriptor(infiniopHandle_t handle,
                                 infiniopTensorDescriptor_t permuted_input_desc,
                                 infiniopTensorDescriptor_t gating_weights_desc,
                                 infiniopTensorDescriptor_t aux_info_desc,
                                 infiniopTensorDescriptor_t output_desc)
        : infiniopDescriptor(handle, INFOP_OP_MOE_COMBINE),
          info(permuted_input_desc, gating_weights_desc, aux_info_desc, output_desc) {}
};

} // namespace infiniop

void infiniopCreateMoECombineDescriptor(
    infiniopHandle_t handle,
    infiniopMoECombineDescriptor_t *desc,
    infiniopTensorDescriptor_t permuted_input_desc,
    infiniopTensorDescriptor_t gating_weights_desc,
    infiniopTensorDescriptor_t aux_info_desc,
    infiniopTensorDescriptor_t output_desc) {
    *desc = new infiniop::infiniopMoECombineDescriptor(handle, permuted_input_desc, gating_weights_desc, aux_info_desc, output_desc);
}

void infiniopDestroyMoECombineDescriptor(infiniopMoECombineDescriptor_t desc) {
    delete desc;
}

void infiniopMoECombine(infiniopMoECombineDescriptor_t desc,
                        void *output,
                        const void *permuted_input,
                        const void *gating_weights,
                        const void *aux_info,
                        infiniopStream_t stream) {
    desc->info.op(output, permuted_input, gating_weights, aux_info, stream);
} 