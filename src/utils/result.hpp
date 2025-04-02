#ifndef __INFINIUTILS_RESULT_H__
#define __INFINIUTILS_RESULT_H__

#include "check.h"
#include <infinicore.h>
#include <variant>

#define CHECK_RESULT(RESULT)    \
    if (!RESULT) {              \
        return RESULT.status(); \
    }

namespace utils {

template <typename T, typename = std::enable_if_t<!std::is_same_v<T, infiniStatus_t>>>
class Result {
    std::variant<infiniStatus_t, T> _result;

public:
    explicit Result(T value) : _result(std::move(value)) {}
    Result(infiniStatus_t status) : _result(status) {
        if (status == INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Result created with success status but value is not set." << std::endl;
            std::abort();
        }
    }

    infiniStatus_t status() const {
        return _result.index() == 0 ? std::get<0>(_result) : INFINI_STATUS_SUCCESS;
    }

    T take() {
        return std::move(std::get<1>(_result));
    }

    operator bool() const {
        return status() == INFINI_STATUS_SUCCESS;
    }

    T *operator->() {
        return &std::get<1>(_result);
    }

    const T *operator->() const {
        return &std::get<1>(_result);
    }

    T &operator*() {
        return std::get<1>(_result);
    }

    const T &operator*() const {
        return std::get<1>(_result);
    }
};

} // namespace utils

#endif // __INFINIUTILS_RESULT_H__
