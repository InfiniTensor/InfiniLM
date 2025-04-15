#ifndef __INFINIOP_ELEMENTWISE_CUDA_API_H__
#define __INFINIOP_ELEMENTWISE_CUDA_API_H__

#include "../elementwise.h"

namespace op::elementwise::cuda {

/**
 * @brief Define the methods and info needed by CUDA to perform elementwise operation
 */
class DeviceImpl final {
    struct Opaque;
    std::shared_ptr<Opaque> _opaque;

    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    ~DeviceImpl() = default;

    template <typename... Args>
    static utils::Result<DeviceImpl *> create(Args &&...args);

    /**
     * @brief Launches elementwise operation where all input types are the same.
     *
     * Calls the corresponding templated `calculateImpl` with a unified input type.
     *
     * @tparam BLOCK_SIZE  Number of threads per block.
     * @tparam Op          Operation functor defining the computation.
     * @tparam Tdata       Data type for both input and output tensors.
     * @tparam Args...     Additional arguments passed to the operation.
     *
     * @param info         Metadata describing tensor shapes, strides, etc.
     * @param workspace    Pointer to workspace buffer on device.
     * @param output       Pointer to output buffer on device.
     * @param inputs       Vector of input pointers (device memory).
     * @param stream       CUDA stream (opaque void*).
     * @param args         Additional operation-specific arguments.
     * @return infiniStatus_t  Status indicating success or failure.
     */
    template <unsigned int BLOCK_SIZE, typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);

    /**
     * @brief Launches elementwise operation where input types may differ.
     *
     * Dispatches to templated `calculateImpl` using specified output and input types.
     *
     * @tparam BLOCK_SIZE  Number of threads per block.
     * @tparam Op          Operation functor defining the computation.
     * @tparam Tout        Output data type.
     * @tparam Tin...      Input data types (must match Op::num_inputs).
     * @tparam Args...     Additional arguments passed to the operation.
     *
     * @param info         Metadata describing tensor shapes, strides, etc.
     * @param workspace    Pointer to workspace buffer on device.
     * @param output       Pointer to output buffer on device.
     * @param inputs       Vector of input pointers (device memory).
     * @param stream       CUDA stream (opaque void*).
     * @param args         (UNUSED) Additional operation-specific arguments.
     * @return infiniStatus_t  Status indicating success or failure.
     */
    template <unsigned int BLOCK_SIZE, typename Op, typename Tout, typename... Tin,
              typename... Args,
              std::enable_if_t<(sizeof...(Tin) == Op::num_inputs), int> = 0>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);
};
} // namespace op::elementwise::cuda

/**
 * @brief Define the process for initializing a Descriptor of an elementwise operation
 * for its CUDA implementation
 *
 * @param HANDLE         The device handle.
 * @param DTYPE          The output dtype.
 * @param OUT_DESC       The output tensor descriptor.
 * @param INPUT_DESC_VEC A vector containing input tensor descriptors.
 */
#define CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)          \
                                                                                             \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC);   \
    CHECK_RESULT(info_result);                                                               \
    auto info = info_result.take();                                                          \
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);      \
                                                                                             \
    auto device_impl_result = op::elementwise::cuda::DeviceImpl::create(HANDLE->internal()); \
    CHECK_RESULT(device_impl_result);                                                        \
                                                                                             \
    *desc_ptr = new Descriptor(                                                              \
        DTYPE,                                                                               \
        std::move(info),                                                                     \
        std::move(device_impl_result.take()),                                                \
        workspace_size,                                                                      \
        HANDLE->device,                                                                      \
        HANDLE->device_id);

#endif // __INFINIOP_ELEMENTWISE_CUDA_API_H__
