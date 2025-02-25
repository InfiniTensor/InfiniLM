#ifndef __ACLNN_TENSOR__
#define __ACLNN_TENSOR__

#include "common_ascend.h"
#include "infiniop/operator.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <aclnn/acl_meta.h>
#include <vector>

// Aclnn tensor descriptor,
// used to build aclTensor
struct aclnnTensorDescriptor {
    uint64_t ndim;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset;
    aclDataType dataType;
    aclFormat format;
    std::vector<int64_t> storageShape;
    int64_t storageNdim;

    aclTensor *t;

    // Transfer from infiniOp DT to aclDataType
    infiniStatus_t setDescriptor(aclDataType dtype, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides);
    infiniStatus_t inferStorageShape();
    // Convert form InfiniOpTensorDescriptor
    infiniStatus_t fromInfiniOpTensorDescriptor(infiniopTensorDescriptor_t y_desc);
    infiniStatus_t createTensor(void *data = nullptr);
    infiniStatus_t destroyTensor();
    ~aclnnTensorDescriptor();

    char *toString();
};

typedef aclnnTensorDescriptor *aclnnTensorDescriptor_t;

#endif
