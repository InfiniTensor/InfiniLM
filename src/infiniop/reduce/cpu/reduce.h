#ifndef __INFINIOP_REDUCE_CPU_H__
#define __INFINIOP_REDUCE_CPU_H__
#include "../../../utils.h"
#include <cstddef>

#ifdef ENABLE_OMP
#include <omp.h>
#endif

#include <type_traits>

namespace op::common_cpu {

namespace reduce_op {

template <typename T>
using ReduceToSame = std::disjunction<
    std::is_same<T, float>,
    std::is_same<T, double>,
    std::is_same<T, uint8_t>,
    std::is_same<T, int8_t>,
    std::is_same<T, uint16_t>,
    std::is_same<T, int16_t>,
    std::is_same<T, uint32_t>,
    std::is_same<T, int32_t>,
    std::is_same<T, uint64_t>,
    std::is_same<T, int64_t>>;

template <typename T, typename = std::enable_if_t<ReduceToSame<T>::value>>
T sum(const T *data, size_t len, ptrdiff_t stride = 1) {
    T result = 0;
    for (size_t i = 0; i < len; i++) {
        result += data[i * stride];
    }

    return result;
}

float sum(const fp16_t *data, size_t len, ptrdiff_t stride = 1) {
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        result += utils::cast<float>(data[i * stride]);
    }

    return result;
}

template <typename T, typename = std::enable_if_t<ReduceToSame<T>::value>>
T sumSquared(const T *data, size_t len, ptrdiff_t stride = 1) {
    T result = 0;
    for (size_t i = 0; i < len; i++) {
        T val = data[i * stride];
        result += val * val;
    }

    return result;
}

float sumSquared(const fp16_t *data, size_t len, ptrdiff_t stride = 1) {
    float result = 0;
    for (size_t i = 0; i < len; i++) {
        float val = utils::cast<float>(data[i * stride]);
        result += val * val;
    }

    return result;
}

} // namespace reduce_op

} // namespace op::common_cpu

#endif //__INFINIOP_REDUCE_CPU_H__
