#ifndef __ACLNN_MATMUL_H__
#define __ACLNN_MATMUL_H__

#include "../../../devices/ascend/tensor_aclnn.h"
#include "../../utils.h"
#include "../blas.h"
#include "matmul_aclnn_api.h"
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/level2/aclnn_gemm.h>

struct InfiniopMatmulAclnnDescriptor {
    infiniDevice_t device;
    int device_id;
    aclOpExecutor *executor;
    MatmulInfo *info;
    infiniDtype_t dtype;
    aclnnTensorDescriptor_t cDesc, aDesc, bDesc;
    // cubeMathType
    // see doc:
    // https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha002/apiref/appdevgapi/context/aclnnBatchMatMul.md
    int8_t mt;
    size_t workspaceSize;

    InfiniopMatmulAclnnDescriptor(infiniDevice_t _device);
};

#endif
