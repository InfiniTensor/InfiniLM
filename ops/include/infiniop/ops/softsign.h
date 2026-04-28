
#ifndef __INFINIOP_SOFTSIGN_API_H__
#define __INFINIOP_SOFTSIGN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSoftsignDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSoftsignDescriptor(infiniopHandle_t handle,
                                                                    infiniopSoftsignDescriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t y,
                                                                    infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetSoftsignWorkspaceSize(infiniopSoftsignDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSoftsign(infiniopSoftsignDescriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *y,
                                                    const void *x,
                                                    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySoftsignDescriptor(infiniopSoftsignDescriptor_t desc);

#endif
