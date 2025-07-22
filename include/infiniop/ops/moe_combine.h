#ifndef __INFINIOP_MOE_COMBINE_API_H__
#define __INFINIOP_MOE_COMBINE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoECombineDescriptor_t;

__C __export infiniStatus_t infiniopCreateMoECombineDescriptor(
    infiniopHandle_t handle,
    infiniopMoECombineDescriptor_t *desc,
    infiniopTensorDescriptor_t permuted_input_desc,
    infiniopTensorDescriptor_t gating_weights_desc,
    infiniopTensorDescriptor_t aux_info_desc,
    infiniopTensorDescriptor_t output_desc);

__C __export infiniStatus_t infiniopDestroyMoECombineDescriptor(infiniopMoECombineDescriptor_t desc);

__C __export infiniStatus_t infiniopMoECombine(infiniopMoECombineDescriptor_t desc,
                        void *output,
                        const void *permuted_input,
                        const void *gating_weights,
                        const void *aux_info,
                        void *stream);

#endif 