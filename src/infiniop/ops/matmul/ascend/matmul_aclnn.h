#ifndef __ACLNN_MATMUL_H__
#define __ACLNN_MATMUL_H__

#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "../blas.h"
#include "operators.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/level2/aclnn_gemm.h>
#include <aclnnop/aclnn_matmul.h>

struct MatmulAclnnDescriptor {
    Device device;
    int device_id;
    aclOpExecutor* executor;
    MatmulInfo* info;
    DT dtype;
    aclnnTensorDescriptor_t cDesc, aDesc, bDesc;
    // cubeMathType
    // see doc: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    float alpha;
    float beta;
    int8_t mt;
    uint64_t workspaceSize;

    MatmulAclnnDescriptor(Device _device);
};

typedef struct MatmulAclnnDescriptor *MatmulAclnnDescriptor_t;

infiniopStatus_t aclnnCreateMatmulDescriptor(AscendHandle_t handle,
                                             MatmulAclnnDescriptor_t *desc_ptr,
                                             infiniopTensorDescriptor_t c_desc,
                                             float alpha,
                                             infiniopTensorDescriptor_t a_desc,
                                             infiniopTensorDescriptor_t b_desc,
                                             float beta,
                                             int8_t cubeMathType);

infiniopStatus_t aclnnGetMatmulWorkspaceSize(MatmulAclnnDescriptor_t desc,
                                             uint64_t *size);

infiniopStatus_t aclnnMatmul(MatmulAclnnDescriptor_t desc,
                             void *workspace,
                             uint64_t workspace_size,
                             void *c,
                             const void *a,
                             const void *b,
                             void *stream);

infiniopStatus_t aclnnDestroyMatmulDescriptor(MatmulAclnnDescriptor_t desc);

#endif
