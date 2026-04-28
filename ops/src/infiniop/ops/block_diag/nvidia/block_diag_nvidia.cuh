#ifndef __BLOCK_DIAG_NVIDIA_H__
#define __BLOCK_DIAG_NVIDIA_H__

#include "../../../operator.h"
#include <vector>

namespace op::block_diag::nvidia {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t num_inputs;
    std::vector<size_t> output_shape;
    ptrdiff_t output_stride0;
    ptrdiff_t output_stride1;
    std::vector<size_t> row_offsets;
    std::vector<size_t> col_offsets;
    std::vector<std::vector<size_t>> input_shapes;
    std::vector<size_t> input_rows;
    std::vector<size_t> input_cols;
    std::vector<ptrdiff_t> input_stride0;
    std::vector<ptrdiff_t> input_stride1;
    size_t output_size;

    Descriptor(infiniDtype_t dtype, size_t num_inputs,
               std::vector<size_t> output_shape,
               ptrdiff_t output_stride0,
               ptrdiff_t output_stride1,
               std::vector<size_t> row_offsets,
               std::vector<size_t> col_offsets,
               std::vector<std::vector<size_t>> input_shapes,
               std::vector<size_t> input_rows,
               std::vector<size_t> input_cols,
               std::vector<ptrdiff_t> input_stride0,
               std::vector<ptrdiff_t> input_stride1,
               size_t output_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          num_inputs(num_inputs),
          output_shape(std::move(output_shape)),
          output_stride0(output_stride0),
          output_stride1(output_stride1),
          row_offsets(std::move(row_offsets)),
          col_offsets(std::move(col_offsets)),
          input_shapes(std::move(input_shapes)),
          input_rows(std::move(input_rows)),
          input_cols(std::move(input_cols)),
          input_stride0(std::move(input_stride0)),
          input_stride1(std::move(input_stride1)),
          output_size(output_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t *input_descs,
        size_t num_inputs);

    size_t workspaceSize() const {
        return 4 * num_inputs * sizeof(size_t) + 2 * num_inputs * sizeof(ptrdiff_t) + num_inputs * sizeof(void *);
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void **inputs,
        void *stream) const;
};

} // namespace op::block_diag::nvidia

#endif // __BLOCK_DIAG_NVIDIA_H__
