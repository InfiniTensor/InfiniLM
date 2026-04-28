#ifndef __BLOCK_DIAG_CPU_H__
#define __BLOCK_DIAG_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <vector>

namespace op::block_diag::cpu {

struct BlockDiagInfo {
    size_t num_inputs;
    std::vector<std::vector<size_t>> input_shapes; // Each input is 2D matrix
    std::vector<ptrdiff_t> input_stride0;          // stride(0) per input
    std::vector<ptrdiff_t> input_stride1;          // stride(1) per input
    std::vector<size_t> output_shape;              // 2D output
    ptrdiff_t output_stride0;
    ptrdiff_t output_stride1;
    std::vector<size_t> row_offsets; // Row offset for each input matrix
    std::vector<size_t> col_offsets; // Column offset for each input matrix
    size_t output_size;

    static utils::Result<BlockDiagInfo> create(
        infiniopTensorDescriptor_t *input_descs,
        size_t num_inputs,
        infiniopTensorDescriptor_t y_desc);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    BlockDiagInfo _info;

    Descriptor(infiniDtype_t dtype, BlockDiagInfo info,
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
        infiniopTensorDescriptor_t *input_descs,
        size_t num_inputs);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void **inputs,
        void *stream) const;
};

} // namespace op::block_diag::cpu

#endif // __BLOCK_DIAG_CPU_H__
