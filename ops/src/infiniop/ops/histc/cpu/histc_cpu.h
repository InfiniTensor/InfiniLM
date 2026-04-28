#ifndef __HISTC_CPU_H__
#define __HISTC_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::histc::cpu {

struct HistcInfo {
    size_t input_size;
    int64_t bins;
    double min_val;
    double max_val;
    ptrdiff_t input_stride;
    ptrdiff_t output_stride;

    static utils::Result<HistcInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        int64_t bins,
        double min_val,
        double max_val);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    HistcInfo _info;

    Descriptor(infiniDtype_t dtype, HistcInfo info,
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
        int64_t bins,
        double min_val,
        double max_val);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::histc::cpu

#endif // __HISTC_CPU_H__
