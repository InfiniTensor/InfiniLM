#ifndef __INFINICORE_API_H__
#define __INFINICORE_API_H__

#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#include <cstddef>
#else
#define __C
#include <stddef.h>
#endif

typedef enum {
    // Success
    INFINI_STATUS_SUCCESS = 0,
    // General Errors
    INFINI_STATUS_INTERNAL_ERROR = 1,
    INFINI_STATUS_NOT_IMPLEMENTED = 2,
    INFINI_STATUS_BAD_PARAM = 3,
    INFINI_STATUS_NULL_POINTER = 4,
    INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED = 5,
    INFINI_STATUS_DEVICE_NOT_FOUND = 6,
    INFINI_STATUS_DEVICE_NOT_INITIALIZED = 7,
    INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED = 8,
    // Op Errors
    INFINI_STATUS_BAD_TENSOR_DTYPE = 10,
    INFINI_STATUS_BAD_TENSOR_SHAPE = 11,
    INFINI_STATUS_BAD_TENSOR_STRIDES = 12,
    INFINI_STATUS_INSUFFICIENT_WORKSPACE = 13,
} infiniStatus_t;

typedef enum {
    INFINI_DEVICE_CPU = 0,
    INFINI_DEVICE_NVIDIA = 1,
    INFINI_DEVICE_CAMBRICON = 2,
    INFINI_DEVICE_ASCEND = 3,
    INFINI_DEVICE_METAX = 4,
    INFINI_DEVICE_MOORE = 5,
    INFINI_DEVICE_ILUVATAR = 6,
    INFINI_DEVICE_KUNLUN = 7,
    INFINI_DEVICE_SUGON = 8,
    INFINI_DEVICE_TYPE_COUNT
} infiniDevice_t;

typedef enum {
    INFINI_DTYPE_INVALID = 0,
    INFINI_DTYPE_BYTE = 1,
    INFINI_DTYPE_BOOL = 2,
    INFINI_DTYPE_I8 = 3,
    INFINI_DTYPE_I16 = 4,
    INFINI_DTYPE_I32 = 5,
    INFINI_DTYPE_I64 = 6,
    INFINI_DTYPE_U8 = 7,
    INFINI_DTYPE_U16 = 8,
    INFINI_DTYPE_U32 = 9,
    INFINI_DTYPE_U64 = 10,
    INFINI_DTYPE_F8 = 11,
    INFINI_DTYPE_F16 = 12,
    INFINI_DTYPE_F32 = 13,
    INFINI_DTYPE_F64 = 14,
    INFINI_DTYPE_C16 = 15,
    INFINI_DTYPE_C32 = 16,
    INFINI_DTYPE_C64 = 17,
    INFINI_DTYPE_C128 = 18,
    INFINI_DTYPE_BF16 = 19,
} infiniDtype_t;

#endif // __INFINICORE_API_H__
