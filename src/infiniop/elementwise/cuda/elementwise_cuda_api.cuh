#ifndef __INFINIOP_ELEMENTWISE_CUDA_API_H__
#define __INFINIOP_ELEMENTWISE_CUDA_API_H__

#include "../elementwise.h"

namespace op::elementwise::cuda {

/**
 * @brief Define the methods and info needed by CUDA to perform elementwise operation
 */
class DeviceImpl final {
    struct Opaque;
    std::shared_ptr<struct Opaque> _opaque;

    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    ~DeviceImpl() = default;

    template <typename... Args>
    static infiniStatus_t create(
        DeviceImpl **device_info,
        Args &&...args);

    /* Invoke elementwise operation when all inputs have the same dtype */
    template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);

    /* Invoke elementwise operation for different input types */
    template <unsigned int BLOCK_SIZE, typename Op, typename Tout, typename... Tin,
              typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);
};
} // namespace op::elementwise::cuda

/**
 * @brief Define the process for initializing a Descriptor of an elementwise operation
 * for its CUDA implementation
 */
#define CREATE_ELEMENTWISE_CUDA_DESCRIPTOR                                                     \
                                                                                               \
    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc);         \
    CHECK_RESULT(info_result);                                                                 \
                                                                                               \
    op::elementwise::cuda::DeviceImpl *device_impl;                                            \
    CHECK_STATUS(op::elementwise::cuda::DeviceImpl::create(&device_impl, handle->internal())); \
                                                                                               \
    *desc_ptr = new Descriptor(                                                                \
        dtype,                                                                                 \
        std::move(info_result.take()),                                                         \
        device_impl,                                                                           \
        handle->device,                                                                        \
        handle->device_id);

#endif // __INFINIOP_ELEMENTWISE_CUDA_API_H__
