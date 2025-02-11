#ifndef __COMMON_ASCEND_H__
#define __COMMON_ASCEND_H__

#include "operators.h"
#include <acl/acl.h>
#include <acl/acl_base.h>
#include <acl/acl_rt.h>
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

#ifdef __cplusplus
};
#endif

int64_t numElements(const int64_t *shape, int64_t num);
const char *dataTypeToString(aclDataType dtype);
const char *formatToString(aclFormat format);
infiniopStatus_t mallocWorkspace(void **workspaceAddr, uint64_t workspaceSize);
infiniopStatus_t freeWorkspace(void *workspaceAddr);
aclDataType toAclDataType(DT dt);

#endif
