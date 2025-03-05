#ifndef __INFINIOP_ADD_API_H__
#define __INFINIOP_ADD_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopAddDescriptor_t;

__C __export infiniStatus_t infiniopCreateAddDescriptor(infiniopHandle_t handle,
                                                        infiniopAddDescriptor_t *desc_ptr,
                                                        infiniopTensorDescriptor_t c,
                                                        infiniopTensorDescriptor_t a,
                                                        infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopAdd(infiniopAddDescriptor_t desc,
                                        void *c,
                                        void const *a,
                                        void const *b,
                                        void *stream);

__C __export infiniStatus_t infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc);

#endif
