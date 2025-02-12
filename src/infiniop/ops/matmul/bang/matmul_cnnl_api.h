#ifndef __CNNL_MATMUL_API_H__
#define __CNNL_MATMUL_API_H__

#include "../../../devices/bang/bang_handle.h"
#include "infiniop/operator.h"

struct InfiniopMatmulBangDescriptor;
typedef struct InfiniopMatmulBangDescriptor *infiniopMatmulBangDescriptor_t;

infiniopStatus_t bangCreateMatmulDescriptor(
    infiniopBangHandle_t handle, infiniopMatmulBangDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc);

infiniopStatus_t bangGetMatmulWorkspaceSize(infiniopMatmulBangDescriptor_t desc,
                                            size_t *size);

infiniopStatus_t bangMatmul(infiniopMatmulBangDescriptor_t desc,
                            void *workspace, size_t workspace_size, void *c,
                            void const *a, void const *b, float alpha,
                            float beta, void *stream);

infiniopStatus_t
bangDestroyMatmulDescriptor(infiniopMatmulBangDescriptor_t desc);

#endif
