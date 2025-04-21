#ifndef INFINIUTILS_CHECK_H
#define INFINIUTILS_CHECK_H
#include <iostream>
#include <tuple>

#define CHECK_OR_RETURN(CONDITION, ERROR)                                    \
    do {                                                                     \
        if (!(CONDITION)) {                                                  \
            std::cerr << "Check Failed: `(" << #CONDITION << ")` is False"   \
                      << " from " << __func__                                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return ERROR;                                                    \
        }                                                                    \
    } while (0)

#define CHECK_API_OR(API, EXPECT, ACTION)                                       \
    do {                                                                        \
        auto api_result_ = (API);                                               \
        if (api_result_ != (EXPECT)) {                                          \
            std::cerr << "Error Code " << api_result_ << " in `" << #API << "`" \
                      << " from " << __func__                                   \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;    \
            { ACTION; }                                                         \
        }                                                                       \
    } while (0)

#define CHECK_INTERNAL(API, EXPECT) CHECK_API_OR(API, EXPECT, return INFINI_STATUS_INTERNAL_ERROR)

#define CHECK_STATUS(API) CHECK_API_OR(API, INFINI_STATUS_SUCCESS, return api_result_)

#define CHECK_DTYPE(DT, ...)                                 \
    do {                                                     \
        auto found_supported_dtype = false;                  \
        for (auto dt : {__VA_ARGS__}) {                      \
            if (dt == DT) {                                  \
                found_supported_dtype = true;                \
                break;                                       \
            }                                                \
        }                                                    \
        CHECK_API_OR(found_supported_dtype, true,            \
                     return INFINI_STATUS_BAD_TENSOR_DTYPE); \
    } while (0)

#define CHECK_DTYPE_ANY_INT(DT)                                                        \
    CHECK_DTYPE(DT,                                                                    \
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64, \
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

#define CHECK_SAME_VEC(ERR, FIRST, ...)              \
    do {                                             \
        for (const auto &shape___ : {__VA_ARGS__}) { \
            if (FIRST != shape___) {                 \
                return ERR;                          \
            }                                        \
        }                                            \
    } while (0)

#define CHECK_SAME_SHAPE(FIRST, ...) CHECK_SAME_VEC(INFINI_STATUS_BAD_TENSOR_SHAPE, FIRST, __VA_ARGS__)

#define CHECK_SAME_STRIDES(FIRST, ...) CHECK_SAME_VEC(INFINI_STATUS_BAD_TENSOR_STRIDES, FIRST, __VA_ARGS__)

#endif // INFINIUTILS_CHECK_H
