#pragma once
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void histc_kernel(
    float *hist,
    const T *input,
    size_t input_size,
    ptrdiff_t input_stride,
    int64_t bins,
    double min_val,
    double max_val) {

    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t stride = static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x);
    size_t input_stride_u = static_cast<size_t>(input_stride);

    double bin_width = (max_val - min_val) / static_cast<double>(bins);

    for (size_t i = idx; i < input_size; i += stride) {
        double val = static_cast<double>(input[i * input_stride_u]);

        // Skip values outside range
        if (val < min_val || val > max_val) {
            continue;
        }

        // Calculate bin index
        int64_t bin_idx = static_cast<int64_t>((val - min_val) / bin_width);

        // Handle edge case: max_val should go to last bin
        if (bin_idx >= bins) {
            bin_idx = bins - 1;
        }
        if (bin_idx < 0) {
            bin_idx = 0;
        }

        atomicAdd(&hist[bin_idx], 1.0f);
    }
}

} // namespace op::cuda
