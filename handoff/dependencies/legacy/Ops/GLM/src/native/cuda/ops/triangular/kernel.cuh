#ifndef INFINI_OPS_CUDA_TRIANGULAR_KERNEL_CUH_
#define INFINI_OPS_CUDA_TRIANGULAR_KERNEL_CUH_

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace infini::ops::triangular_detail {

constexpr int kMaxRank = 8;

struct Metadata {
  int rank{};
  std::size_t shape[kMaxRank]{};
  std::int64_t input_strides[kMaxRank]{};
  std::int64_t output_strides[kMaxRank]{};
  std::size_t rows{};
  std::size_t columns{};
  std::size_t elements{};
  std::int64_t diagonal{};
};

template <typename T, bool kUpper>
__global__ void TriangularKernel(const T* input, T* output, Metadata meta) {
  const std::size_t step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
       index < meta.elements; index += step) {
    const std::size_t column = index % meta.columns;
    const std::size_t row = index / meta.columns % meta.rows;
    std::int64_t input_offset = 0;
    std::int64_t output_offset = 0;
    if (meta.rank <= 3) {
      const std::size_t batch = meta.rank == 3
                                    ? index / (meta.columns * meta.rows)
                                    : 0;
      input_offset = static_cast<std::int64_t>(row) *
                         meta.input_strides[meta.rank - 2] +
                     static_cast<std::int64_t>(column) *
                         meta.input_strides[meta.rank - 1];
      output_offset = static_cast<std::int64_t>(row) *
                          meta.output_strides[meta.rank - 2] +
                      static_cast<std::int64_t>(column) *
                          meta.output_strides[meta.rank - 1];
      if (meta.rank == 3) {
        input_offset += static_cast<std::int64_t>(batch) *
                        meta.input_strides[0];
        output_offset += static_cast<std::int64_t>(batch) *
                         meta.output_strides[0];
      }
    } else {
      std::size_t remaining = index;
      for (int dimension = meta.rank - 1; dimension >= 0; --dimension) {
        const std::size_t coordinate = remaining % meta.shape[dimension];
        remaining /= meta.shape[dimension];
        input_offset += static_cast<std::int64_t>(coordinate) *
                        meta.input_strides[dimension];
        output_offset += static_cast<std::int64_t>(coordinate) *
                         meta.output_strides[dimension];
      }
    }
    const std::int64_t relative = static_cast<std::int64_t>(column) -
                                  static_cast<std::int64_t>(row);
    const bool keep = kUpper ? relative >= meta.diagonal
                             : relative <= meta.diagonal;
    output[output_offset] = keep ? input[input_offset] : T{};
  }
}

}  // namespace infini::ops::triangular_detail

#endif  // INFINI_OPS_CUDA_TRIANGULAR_KERNEL_CUH_
