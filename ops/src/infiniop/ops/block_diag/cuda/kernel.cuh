#include <cstddef>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void block_diag_kernel(
    T *output,
    const T **inputs,
    size_t num_inputs,
    size_t output_rows,
    size_t output_cols,
    ptrdiff_t output_stride0,
    ptrdiff_t output_stride1,
    const size_t *row_offsets,
    const size_t *col_offsets,
    const size_t *input_rows,
    const size_t *input_cols,
    const ptrdiff_t *input_stride0,
    const ptrdiff_t *input_stride1) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = output_rows * output_cols;

    if (idx >= total) {
        return;
    }

    size_t out_row = idx / output_cols;
    size_t out_col = idx % output_cols;

    const ptrdiff_t out_offset = static_cast<ptrdiff_t>(out_row) * output_stride0 + static_cast<ptrdiff_t>(out_col) * output_stride1;

    T value = T{};

    // Find which input matrix this output position belongs to
    for (size_t i = 0; i < num_inputs; ++i) {
        size_t row_start = row_offsets[i];
        size_t row_end = row_start + input_rows[i];
        size_t col_start = col_offsets[i];
        size_t col_end = col_start + input_cols[i];

        if (out_row >= row_start && out_row < row_end && out_col >= col_start && out_col < col_end) {
            // This position belongs to input i
            size_t in_row = out_row - row_start;
            size_t in_col = out_col - col_start;
            const ptrdiff_t in_offset = static_cast<ptrdiff_t>(in_row) * input_stride0[i] + static_cast<ptrdiff_t>(in_col) * input_stride1[i];
            value = inputs[i][in_offset];
            break;
        }
    }
    output[out_offset] = value;
}

} // namespace op::cuda
