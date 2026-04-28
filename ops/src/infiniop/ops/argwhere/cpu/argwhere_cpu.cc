#include "argwhere_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "infinicore.h"
#include <array>
#include <cstddef>
#include <cstdint>
namespace op::argwhere::cpu {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = ArgwhereInfo::create(x_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateArgWhere(
    const ArgwhereInfo &info,
    void *workspace,
    size_t workspace_size,
    void **y,
    size_t *count,
    const void *x) {

    const Tdata *x_data = reinterpret_cast<const Tdata *>(x);

    std::vector<int64_t> positions;
    const size_t ndim = info.shapes.size();

    for (size_t i = 0; i < info.num_elements; i++) {
        size_t pos = 0;
        size_t tmp = i;

        std::vector<int64_t> coord(ndim);

        // unravel index
        for (size_t j = ndim; j-- > 0;) {
            coord[j] = tmp % info.shapes[j];
            tmp /= info.shapes[j];
            pos += coord[j] * info.strides[j];
        }

        // PyTorch semantics: != 0
        if (x_data[pos] != Tdata(0)) {
            for (size_t j = 0; j < ndim; j++) {
                positions.push_back(coord[j]);
            }
        }
    }

    *count = positions.size() / ndim;

    *y = new int64_t[positions.size()];
    memcpy(*y, positions.data(), positions.size() * sizeof(int64_t));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void **y,
    size_t *count,
    const void *x,
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_F32:
        return calculateArgWhere<float>(_info, workspace, workspace_size, y, count, x);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::argwhere::cpu
