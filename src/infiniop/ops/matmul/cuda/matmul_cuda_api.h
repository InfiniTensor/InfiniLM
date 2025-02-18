#ifndef __INFINIOP_MATMUL_CUDA_API_H__
#define __INFINIOP_MATMUL_CUDA_API_H__

#include "../../../devices/cuda/cuda_handle.h"
#include "infiniop/operator.h"

struct InfiniopMatmulCudaDescriptor;
typedef struct InfiniopMatmulCudaDescriptor *infiniopMatmulCudaDescriptor_t;

infiniopStatus_t cudaCreateMatmulDescriptor(infiniopCudaHandle_t handle,
                                            infiniopMatmulCudaDescriptor_t *desc_ptr,
                                            infiniopTensorDescriptor_t c_desc,
                                            infiniopTensorDescriptor_t a_desc,
                                            infiniopTensorDescriptor_t b_desc);

infiniopStatus_t cudaGetMatmulWorkspaceSize(infiniopMatmulCudaDescriptor_t desc, size_t *size);

infiniopStatus_t cudaMatmul(infiniopMatmulCudaDescriptor_t desc,
                            void *workspace,
                            size_t workspace_size,
                            void *c,
                            void const *a,
                            void const *b,
                            float alpha,
                            float beta,
                            void *stream);

infiniopStatus_t cudaDestroyMatmulDescriptor(infiniopMatmulCudaDescriptor_t desc);

#endif // __INFINIOP_MATMUL_CUDA_API_H__
