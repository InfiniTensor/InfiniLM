#ifndef __INFINIOP_ELEMENTWISE_BANG_API_H__
#define __INFINIOP_ELEMENTWISE_BANG_API_H__

#include "../elementwise.h"

namespace op::elementwise::bang {

/**
 * @brief BANG device implementation for elementwise operations.
 *
 * Provides interface for creating and executing elementwise operations on BANG devices.
 */
class DeviceImpl final {
    struct Opaque;
    std::shared_ptr<Opaque> _opaque;

    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    ~DeviceImpl() = default;

    /**
     * @brief Creates a DeviceImpl instance.
     *
     * @tparam Args Argument types for construction.
     * @param args Arguments forwarded to implementation.
     * @return utils::Result<DeviceImpl*> Result containing new instance.
     */
    template <typename... Args>
    static utils::Result<DeviceImpl *> create(Args &&...args);

    /**
     * @brief Executes elementwise operation on BANG device.
     *
     * @tparam Op       Operator functor type.
     * @tparam Tdata    Data type for inputs and output.
     * @tparam Args     Additional arguments for the operator.
     *
     * @param info      Elementwise operation metadata.
     * @param workspace Device workspace memory.
     * @param output    Output tensor buffer.
     * @param inputs    Vector of input tensor pointers.
     * @param queue    BANG queue (as void*).
     * @param args      Additional arguments for the operator.
     * @return infiniStatus_t Status indicating success or failure.
     */
    template <typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        void *output,
        const std::vector<const void *> &inputs,
        void *queue,
        Args &&...args);
};
} // namespace op::elementwise::bang

/**
 * @brief Macro for creating BANG elementwise operation descriptor.
 *
 * @param HANDLE         Device handle.
 * @param DTYPE          Output data type.
 * @param OUT_DESC       Output tensor descriptor.
 * @param INPUT_DESC_VEC Vector of input tensor descriptors.
 */
#define CREATE_ELEMENTWISE_BANG_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)          \
                                                                                             \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC);   \
    CHECK_RESULT(info_result);                                                               \
    auto info = info_result.take();                                                          \
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);      \
                                                                                             \
    auto device_impl_result = op::elementwise::bang::DeviceImpl::create(HANDLE->internal()); \
    CHECK_RESULT(device_impl_result);                                                        \
                                                                                             \
    *desc_ptr = new Descriptor(                                                              \
        DTYPE,                                                                               \
        std::move(info),                                                                     \
        std::move(device_impl_result.take()),                                                \
        workspace_size,                                                                      \
        HANDLE->device,                                                                      \
        HANDLE->device_id);

#endif // __INFINIOP_ELEMENTWISE_BANG_API_H__
