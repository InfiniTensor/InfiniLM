#ifndef __LOGDET_CPU_H__
#define __LOGDET_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <cstddef>
#include <vector>

namespace op::logdet::cpu {

struct LogdetInfo {
    size_t matrix_size; // N x N matrix
    size_t input_size;
    std::vector<ptrdiff_t> input_strides;

    static utils::Result<LogdetInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    LogdetInfo _info;

    Descriptor(infiniDtype_t dtype, LogdetInfo info,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc);

    size_t workspaceSize() const {
        const size_t elem_size = (_dtype == INFINI_DTYPE_F32) ? sizeof(float) : sizeof(double);
        return _info.matrix_size * _info.matrix_size * elem_size;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::logdet::cpu

#endif // __LOGDET_CPU_H__
