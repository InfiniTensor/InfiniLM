#ifndef __INFINIOP_LDEXP_API_H__
#define __INFINIOP_LDEXP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLdexpDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateLdexpDescriptor(infiniopHandle_t handle,
                                                                 infiniopLdexpDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x,
                                                                 infiniopTensorDescriptor_t exp);
__INFINI_C __export infiniStatus_t infiniopGetLdexpWorkspaceSize(infiniopLdexpDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLdexp(infiniopLdexpDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 const void *exp,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLdexpDescriptor(infiniopLdexpDescriptor_t desc);

#endif // __INFINIOP_LDEXP_API_H__
