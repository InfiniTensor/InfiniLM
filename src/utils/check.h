#ifndef INFINIUTILS_CHECK_H
#define INFINIUTILS_CHECK_H
#include <iostream>

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

#endif // INFINIUTILS_CHECK_H
