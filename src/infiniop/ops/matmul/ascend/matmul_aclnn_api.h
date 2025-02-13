#ifndef __INFINIOP_MATMUL_ACLNN_API_H__
#define __INFINIOP_MATMUL_ACLNN_API_H__
#include "../../../devices/ascend/ascend_handle.h"
#include "infiniop/operator.h"

struct InfiniopMatmulAclnnDescriptor;
typedef struct InfiniopMatmulAclnnDescriptor *MatmulAclnnDescriptor_t;

infiniopStatus_t aclnnCreateMatmulDescriptor(infiniopAscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             int8_t cubeMathType);

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             size_t *size);

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc, void *workspace,
                             size_t workspace_size, void *c, const void *a,
                             const void *b, float alpha, float beta,
                             void *stream);

infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc);
#endif // __INFINIOP_MATMUL_ACLNN_API_H__
