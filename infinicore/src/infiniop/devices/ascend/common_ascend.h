#ifndef __INFINIOP_COMMON_ASCEND_H__
#define __INFINIOP_COMMON_ASCEND_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "ascend_handle.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <cstdio>
#include <functional>
#include <inttypes.h>
#include <numeric>
#include <sstream>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif
#define CHECK_ACL(API) CHECK_INTERNAL(API, ACL_SUCCESS)
#ifdef __cplusplus
};
#endif

struct aclnnTensorDescriptor {
    uint64_t ndim;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    int64_t offset = 0;
    aclDataType dataType;
    aclFormat format;
    std::vector<int64_t> storageShape;
    int64_t storageNdim = 1;
    aclTensor *tensor;

    aclnnTensorDescriptor(aclDataType dtype, const std::vector<int64_t> &shape, const std::vector<int64_t> &strides, void *data = nullptr);
    aclnnTensorDescriptor(infiniopTensorDescriptor_t y_desc, void *data = nullptr);
    ~aclnnTensorDescriptor();
    size_t numel() const;

    std::string toString();
};
typedef aclnnTensorDescriptor *aclnnTensorDescriptor_t;

aclDataType toAclDataType(infiniDtype_t dt);

#define GetRecentErrMsg()                                   \
    {                                                       \
        auto tmp_err_msg = aclGetRecentErrMsg();            \
        if (tmp_err_msg != NULL) {                          \
            printf(" ERROR Message : %s \n ", tmp_err_msg); \
        }                                                   \
    }

#endif
