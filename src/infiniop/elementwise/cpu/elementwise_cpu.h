#ifndef __INFINIOP_ELEMENTWISE_CPU_H__
#define __INFINIOP_ELEMENTWISE_CPU_H__

#include "../../devices/cpu/common_cpu.h"
#include "../elementwise.h"
#include <utility>

/**
 * @brief Define the process for initializing a Descriptor of an elementwise operation
 * for its CPU implementation
 */
#define CREATE_ELEMENTWISE_CPU_DESCRIPTOR                                                         \
                                                                                                  \
    op::elementwise::ElementwiseInfo elementwise_info;                                            \
    CHECK_STATUS(op::elementwise::createElementwiseInfo(elementwise_info, out_desc, input_desc)); \
                                                                                                  \
    *desc_ptr = new Descriptor(                                                                   \
        dtype,                                                                                    \
        std::move(elementwise_info),                                                              \
        nullptr,                                                                                  \
        handle->device,                                                                           \
        handle->device_id);

DEVICE_IMPL(cpu)

namespace op::elementwise::cpu {

struct DeviceImpl::Opaque {};

template <typename... Args>
infiniStatus_t DeviceImpl::create(DeviceImpl **device_info, Args &&...args) {
    *device_info = new DeviceImpl(nullptr);
    return INFINI_STATUS_SUCCESS;
}

// Perform elementwise operation for different input types
template <typename Op, typename Tout, typename... Tin, size_t... Is, typename... Args, std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
void calculate_impl(const op::elementwise::ElementwiseInfo &info, void *output, const std::vector<const void *> &inputs, std::index_sequence<Is...>, Args &&...args) {
    Tout *out = reinterpret_cast<Tout *>(output);
    std::tuple<const Tin *...> input_ptrs = {reinterpret_cast<const Tin *>(inputs[Is])...};
    ptrdiff_t output_size = info.output_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        size_t out_idx = info.output_contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.output_shape.data(), info.output_strides.data());

        auto get_input_idx = [&](size_t input_id) {
            return info.input_contiguous[input_id] ? i
                                                   : (info.input_broadcasted[input_id]
                                                          ? op::common_cpu::indexToReducedOffset(i, info.ndim, info.output_strides.data(), info.input_strides[input_id].data())
                                                          : op::common_cpu::indexToOffset(i, info.ndim, info.input_shapes[input_id].data(), info.input_strides[input_id].data()));
        };

        out[out_idx] = utils::cast<Tout>(Op{}(std::get<Is>(input_ptrs)[get_input_idx(Is)]..., std::forward<Args>(args)...));
    }
}

// Invoke elementwise operation for different input types
template <typename Op, typename Tout, typename... Tin, typename... Args, std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
void DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                           void *output,
                           const std::vector<const void *> &inputs,
                           Args &&...args) {

    static_assert(sizeof...(Tin) == Op::num_inputs, "Input type count mismatch");
    calculate_impl<Op, Tout, Tin...>(info, output, inputs, std::make_index_sequence<sizeof...(Tin)>{}, std::forward<Args>(args)...);
}

// Perform elementwise operation when all inputs have the same type
template <typename Op, typename Tdata, size_t... Is, typename... Args>
void calculate_impl(const op::elementwise::ElementwiseInfo &info,
                    void *output,
                    const std::vector<const void *> &inputs,
                    std::index_sequence<Is...>,
                    Args &&...args) {

    Tdata *out = reinterpret_cast<Tdata *>(output);
    std::array<const Tdata *, sizeof...(Is)> ins = {reinterpret_cast<const Tdata *>(inputs[Is])...};
    const ptrdiff_t output_size = info.output_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        size_t out_idx = info.output_contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.output_shape.data(), info.output_strides.data());

        auto get_input_idx = [&](size_t input_id) {
            return info.input_contiguous[input_id] ? i
                                                   : (info.input_broadcasted[input_id]
                                                          ? op::common_cpu::indexToReducedOffset(i, info.ndim, info.output_strides.data(), info.input_strides[input_id].data())
                                                          : op::common_cpu::indexToOffset(i, info.ndim, info.input_shapes[input_id].data(), info.input_strides[input_id].data()));
        };

        if constexpr (std::is_same_v<Tdata, fp16_t>) {
            out[out_idx] = utils::cast<fp16_t>(Op{}(utils::cast<float>(ins[Is][get_input_idx(Is)])..., std::forward<Args>(args)...));
        } else {
            out[out_idx] = Op{}(ins[Is][get_input_idx(Is)]..., std::forward<Args>(args)...);
        }
    }
}

// Invoke elementwise operation when all inputs have the same type
template <typename Op, typename Tdata, typename... Args>
void DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info, void *output, const std::vector<const void *> &inputs, Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    calculate_impl<Op, Tdata>(info, output, inputs, std::make_index_sequence<N>{}, std::forward<Args>(args)...);
}

} // namespace op::elementwise::cpu

#endif // __INFINIOP_ELEMENTWISE_CPU_H__
