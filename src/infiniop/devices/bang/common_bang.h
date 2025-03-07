#ifndef __COMMON_BANG_H__
#define __COMMON_BANG_H__

#include "../../../utils.h"

// the maximum NRAM memory is 1024 * 768
#define NRAM_MAX_SIZE (1024 * 256)

#define GDRAM_MAX_SIZE (1024 * 1024 * 1024)

#ifdef __cplusplus
extern "C" {
#endif
#define CHECK_BANG(API) CHECK_INTERNAL(API, CNNL_STATUS_SUCCESS)
#ifdef __cplusplus
};
#endif

#endif // __COMMON_BANG_H__
