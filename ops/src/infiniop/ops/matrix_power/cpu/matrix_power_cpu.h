#ifndef __MATRIX_POWER_CPU_H__
#define __MATRIX_POWER_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <vector>

namespace op::matrix_power::cpu {

struct MatrixPowerInfo {
    size_t matrix_size; // N x N matrix
    size_t n;           // Power
    size_t input_size;
    size_t output_size;

    static utils::Result<MatrixPowerInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        int n);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    MatrixPowerInfo _info;

    Descriptor(infiniDtype_t dtype, MatrixPowerInfo info,
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
        infiniopTensorDescriptor_t x_desc,
        int n);

    size_t workspaceSize() const {
        if (_info.n == 0 || _info.matrix_size == 0) {
            return 0;
        }
        const size_t elems = 2 * _info.matrix_size * _info.matrix_size;
        switch (_dtype) {
        case INFINI_DTYPE_F16:
        case INFINI_DTYPE_BF16:
        case INFINI_DTYPE_F32:
            return elems * sizeof(float);
        case INFINI_DTYPE_F64:
            return elems * sizeof(double);
        default:
            return 0;
        }
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::matrix_power::cpu

#endif // __MATRIX_POWER_CPU_H__
