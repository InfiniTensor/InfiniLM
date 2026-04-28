#include "logcumsumexp_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>

#include "../../../../utils/custom_types.h"

namespace op::logcumsumexp::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis,
    int exclusive,
    int reverse) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 注意：这里复用了你之前修改过的 Info 类，它现在包含正确的 stride 信息
    auto result = LogCumSumExpInfo::create(y_desc, x_desc, axis, exclusive, reverse);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void calculate_cpu_impl(
    const LogCumSumExpInfo &info,
    void *y,
    const void *x) {

    size_t outer_size = info.outer_size();
    size_t axis_size = info.axis_size();
    size_t inner_size = info.inner_size();
    bool exclusive = info.exclusive();
    bool reverse = info.reverse();

    size_t x_outer_stride = info._x_outer_stride;
    size_t x_axis_stride = info._x_axis_stride;
    size_t x_inner_stride = info._x_inner_stride;

    size_t y_outer_stride = info._y_outer_stride;
    size_t y_axis_stride = info._y_axis_stride;
    size_t y_inner_stride = info._y_inner_stride;

    auto y_ptr = reinterpret_cast<T *>(y);
    auto x_ptr = reinterpret_cast<const T *>(x);

    // Flatten outer+inner for OpenMP parallelism on Windows
    size_t total_outer_inner = outer_size * inner_size;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t idx = 0; idx < (ptrdiff_t)total_outer_inner; ++idx) {
        // Recover original i and j
        size_t i = idx / inner_size;
        size_t j = idx % inner_size;

        size_t x_base = i * x_outer_stride + j * x_inner_stride;
        size_t y_base = i * y_outer_stride + j * y_inner_stride;

        double running_max = -std::numeric_limits<double>::infinity();
        double running_sum_exp = 0.0;

        for (size_t k = 0; k < axis_size; ++k) {
            size_t k_idx = reverse ? (axis_size - 1 - k) : k;

            size_t x_offset = x_base + k_idx * x_axis_stride;
            size_t y_offset = y_base + k_idx * y_axis_stride;

            float val = utils::cast<float>(x_ptr[x_offset]);

            if (exclusive) {
                if (running_sum_exp == 0.0) {
                    y_ptr[y_offset] = utils::cast<T>(-std::numeric_limits<float>::infinity());
                } else {
                    y_ptr[y_offset] = utils::cast<T>(
                        static_cast<float>(running_max + std::log(running_sum_exp)));
                }
            }

            if (val > running_max) {
                running_sum_exp = running_sum_exp * std::exp(running_max - val) + 1.0;
                running_max = val;
            } else {
                running_sum_exp += std::exp(val - running_max);
            }

            if (!exclusive) {
                y_ptr[y_offset] = utils::cast<T>(
                    static_cast<float>(running_max + std::log(running_sum_exp)));
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, y, x);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, y, x);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, y, x);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logcumsumexp::cpu
