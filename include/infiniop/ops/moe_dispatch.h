#ifndef __INFINIOP_MOE_DISPATCH_API_H__
#define __INFINIOP_MOE_DISPATCH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMoEDispatchDescriptor_t;

__C __export infiniStatus_t infiniopCreateMoEDispatchDescriptor(
    infiniopHandle_t handle,
    infiniopMoEDispatchDescriptor_t *desc,
    int num_experts,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t permuted_output_desc,
    infiniopTensorDescriptor_t aux_info_desc);

__C __export infiniStatus_t infiniopDestroyMoEDispatchDescriptor(infiniopMoEDispatchDescriptor_t desc);

__C __export infiniStatus_t infiniopMoEDispatch(infiniopMoEDispatchDescriptor_t desc,
					   const void *input,
                       const void *indices,
                       void *permuted_output,
                       void *aux_info,
                       void *stream);

#endif 