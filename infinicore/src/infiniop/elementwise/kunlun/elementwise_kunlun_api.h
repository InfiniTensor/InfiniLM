#ifndef __INFINIOP_ELEMENTWISE_KUNLUN_API_H__
#define __INFINIOP_ELEMENTWISE_KUNLUN_API_H__

#include "../elementwise.h"

namespace op::elementwise::kunlun {

class DeviceImpl final {
    struct Opaque;
    std::shared_ptr<Opaque> _opaque;

    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    ~DeviceImpl() = default;

    template <typename... Args>
    static utils::Result<DeviceImpl *> create(Args &&...args);

    template <typename Op, typename Tdata, typename... Args>
    infiniStatus_t calculate(
        const op::elementwise::ElementwiseInfo &info,
        void *workspace,
        void *output,
        const std::vector<const void *> &inputs,
        void *stream,
        Args &&...args);
};

} // namespace op::elementwise::kunlun

#define CREATE_ELEMENTWISE_KUNLUN_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC)          \
                                                                                               \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC);     \
    CHECK_RESULT(info_result);                                                                 \
    auto info = info_result.take();                                                            \
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);        \
                                                                                               \
    auto device_impl_result = op::elementwise::kunlun::DeviceImpl::create(HANDLE->internal()); \
    CHECK_RESULT(device_impl_result);                                                          \
                                                                                               \
    *desc_ptr = new Descriptor(                                                                \
        DTYPE,                                                                                 \
        std::move(info),                                                                       \
        std::move(device_impl_result.take()),                                                  \
        workspace_size,                                                                        \
        HANDLE->device,                                                                        \
        HANDLE->device_id);

#endif
