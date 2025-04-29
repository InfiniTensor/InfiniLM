#ifndef __INFINIOP_ELEMENTWISE_CPU_H__
#define __INFINIOP_ELEMENTWISE_CPU_H__

#include "../../devices/cpu/common_cpu.h"
#include "../elementwise.h"
#include <utility>

/**
 * @brief Define the process for initializing a Descriptor of an elementwise operation
 * for its CPU implementation
 *
 * @param HANDLE         The device handle.
 * @param DTYPE          The output dtype.
 * @param OUT_DESC       The output tensor descriptor.
 * @param INPUT_DESC_VEC A vector containing input tensor descriptors.
 */
#define CREATE_ELEMENTWISE_CPU_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)         \
                                                                                           \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC); \
    CHECK_RESULT(info_result);                                                             \
                                                                                           \
    *desc_ptr = new Descriptor(                                                            \
        DTYPE,                                                                             \
        info_result.take(),                                                                \
        nullptr,                                                                           \
        0,                                                                                 \
        HANDLE->device,                                                                    \
        HANDLE->device_id);

namespace op::elementwise::cpu {

/**
 * @brief CPU-specific device implementation for resource management and
 * calculation implementations.
 *
 * This class encapsulates device-specific behavior and execution logic.
 * Use the static create() method to instantiate a DeviceImpl.
 */
class DeviceImpl final {
    struct Opaque;
    std::shared_ptr<Opaque> _opaque;

    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    ~DeviceImpl() = default;

    template <typename... Args>
    static utils::Result<DeviceImpl> create(Args &&...args);

    /**
     * @brief Dispatches an elementwise operation with uniform input types.
     *
     * @tparam Op   The elementwise operation to perform.
     * @tparam Tdata The common data type of all inputs and output.
     * @tparam Args  Additional backend-specific arguments.
     * @param info     Precomputed tensor metadata (shapes, strides, etc.).
     * @param output   Pointer to the output tensor buffer.
     * @param inputs   Vector of input tensor data pointers.
     * @param stream   Device execution stream.
     * @param args     Additional backend-specific arguments.
     * @return infiniStatus_t  Status indicating success or failure.
     */
    template <typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);

    /**
     * @brief Dispatches an elementwise operation with heterogeneous input types.
     *
     * Supports operations where each input may have a different type, as defined by Op.
     * The number of input types must match the operation's expected input count.
     *
     * @tparam Op     The elementwise operation to perform.
     * @tparam Tout   Output data type.
     * @tparam Tin    Variadic input data types.
     * @tparam Args   Additional backend-specific arguments.
     * @param info     Precomputed tensor metadata (shapes, strides, etc.).
     * @param output   Pointer to the output tensor buffer.
     * @param inputs   Vector of input tensor data pointers.
     * @param stream   Device execution stream.
     * @param args     Additional backend-specific arguments.
     * @return infiniStatus_t  Status indicating success or failure.
     */
    template <typename Op, typename Tout, typename... Tin,
              typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);
};

// Define the Opaque struct for CPU, which is empty
struct DeviceImpl::Opaque {};

template <typename... Args>
utils::Result<DeviceImpl> DeviceImpl::create(Args &&...args) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

// Perform elementwise operation for different input types
template <typename Op, typename Tout, typename... Tin, size_t... Is, typename... Args,
          std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
void calculate_impl(const op::elementwise::ElementwiseInfo &info,
                    void *output,
                    const std::vector<const void *> &inputs,
                    std::index_sequence<Is...>,
                    Args &&...args) {

    Tout *out = reinterpret_cast<Tout *>(output);
    std::tuple<const Tin *...> input_ptrs = {reinterpret_cast<const Tin *>(inputs[Is])...};
    ptrdiff_t output_size = info.getOutputSize();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        size_t out_idx = info.isOutputContiguous()
                           ? i
                           : op::common_cpu::indexToOffset(i, info.getNdim(), info.getOutputShape(), info.getOutputStrides());

        auto get_input_idx = [&](size_t input_id) {
            return info.getInputContiguous()[input_id]
                     ? i
                     : (info.getInputBroadcasted()[input_id]
                            ? op::common_cpu::indexToReducedOffset(i, info.getNdim(), info.getOutputStrides(), info.getInputStrides(input_id))
                            : op::common_cpu::indexToOffset(i, info.getNdim(), info.getInputShape(input_id), info.getInputStrides(input_id)));
        };

        out[out_idx] = utils::cast<Tout>(
            Op{}.template operator()<Tout, Tin...>(std::get<Is>(input_ptrs)[get_input_idx(Is)]..., std::forward<Args>(args)...));
    }
}

// Invoke elementwise operation for different input types
template <typename Op, typename Tout, typename... Tin, typename... Args, std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int>>
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {

    static_assert(sizeof...(Tin) == Op::num_inputs, "Input type count mismatch");
    calculate_impl<Op, Tout, Tin...>(info, output, inputs, std::make_index_sequence<sizeof...(Tin)>{}, std::forward<Args>(args)...);
    return INFINI_STATUS_SUCCESS;
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
    const ptrdiff_t output_size = info.getOutputSize();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        size_t out_idx = info.isOutputContiguous()
                           ? i
                           : op::common_cpu::indexToOffset(i, info.getNdim(), info.getOutputShape(), info.getOutputStrides());

        auto get_input_idx = [&](size_t input_id) {
            return info.getInputContiguous()[input_id]
                     ? i
                     : (info.getInputBroadcasted()[input_id]
                            ? op::common_cpu::indexToReducedOffset(i, info.getNdim(), info.getOutputStrides(), info.getInputStrides(input_id))
                            : op::common_cpu::indexToOffset(i, info.getNdim(), info.getInputShape(input_id), info.getInputStrides(input_id)));
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
infiniStatus_t DeviceImpl::calculate(const op::elementwise::ElementwiseInfo &info,
                                     void *output,
                                     const std::vector<const void *> &inputs,
                                     void *stream,
                                     Args &&...args) {
    constexpr size_t N = Op::num_inputs;
    calculate_impl<Op, Tdata>(info, output, inputs, std::make_index_sequence<N>{}, std::forward<Args>(args)...);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::elementwise::cpu

#endif // __INFINIOP_ELEMENTWISE_CPU_H__
