#ifndef __INFINIOP_COMMON_ASCEND_H__
#define __INFINIOP_COMMON_ASCEND_H__

#include "ascend_handle.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
#include <aclnn/acl_meta.h>
#include <cstdio>
#include <functional>
#include <inttypes.h>
#include <numeric>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define LOG_ERROR(message, ...)              \
    do {                                     \
        printf(message, ##__VA_ARGS__);      \
        return INFINI_STATUS_INTERNAL_ERROR; \
    } while (0)

#ifdef __cplusplus
};
#endif

struct InfiniopAscendHandle {
    infiniDevice_t device;
    int device_id;
};

int64_t numElements(const int64_t *shape, int64_t num);
const char *dataTypeToString(aclDataType dtype);
const char *formatToString(aclFormat format);
infiniStatus_t mallocWorkspace(void **workspaceAddr, size_t workspaceSize);
infiniStatus_t freeWorkspace(void *workspaceAddr);
aclDataType toAclDataType(infiniDtype_t dt);

#endif
