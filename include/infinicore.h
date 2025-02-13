#ifndef __INFINICORE_H__
#define __INFINICORE_H__
#include <stdint.h>

#ifndef __INFINICORE_EXPORT_C__
#define __INFINICORE_EXPORT_C__
#if defined(_WIN32)
#define __export __declspec(dllexport)
#elif defined(__GNUC__) &&                                                     \
    ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define __export __attribute__((visibility("default")))
#else
#define __export
#endif

#ifdef __cplusplus
#define __C extern "C"
#include <cstddef>
#else
#define __C
#include <stddef>
#endif
#endif // __INFINICORE_EXPORT_C__

#ifndef __INFINI_DEVICE__
#define __INFINI_DEVICE__
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
} infiniDevice_t;
#endif // __INFINI_DEVICE__

#ifndef __INFINI_DTYPE__
#define __INFINI_DTYPE__
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
    INFINI_DTYPE_C8 = 15,
    INFINI_DTYPE_C16 = 16,
    INFINI_DTYPE_C32 = 17,
    INFINI_DTYPE_C64 = 18,
    INFINI_DTYPE_BF16 = 19,
} infiniDtype_t;

inline size_t infiniSizeof(infiniDtype_t dtype) {
    switch (dtype) {
    case INFINI_DTYPE_INVALID:
        return 0;
    case INFINI_DTYPE_BYTE:
        return 1;
    case INFINI_DTYPE_BOOL:
        return 1;
    case INFINI_DTYPE_I8:
        return 1;
    case INFINI_DTYPE_I16:
        return 2;
    case INFINI_DTYPE_I32:
        return 4;
    case INFINI_DTYPE_I64:
        return 8;
    case INFINI_DTYPE_U8:
        return 1;
    case INFINI_DTYPE_U16:
        return 2;
    case INFINI_DTYPE_U32:
        return 4;
    case INFINI_DTYPE_U64:
        return 8;
    case INFINI_DTYPE_F8:
        return 1;
    case INFINI_DTYPE_F16:
        return 2;
    case INFINI_DTYPE_F32:
        return 4;
    case INFINI_DTYPE_F64:
        return 8;
    case INFINI_DTYPE_C8:
        return 2;
    case INFINI_DTYPE_C16:
        return 4;
    case INFINI_DTYPE_C32:
        return 8;
    case INFINI_DTYPE_C64:
        return 16;
    case INFINI_DTYPE_BF16:
        return 2;
    default:
        return 0;
    }
}
#endif // __INFINI_DTYPE__

#endif // __INFINICORE_H__
