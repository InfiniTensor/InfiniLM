#ifndef INFINI_OPS_CUDA_TRIANGULAR_KERNEL_H_
#define INFINI_OPS_CUDA_TRIANGULAR_KERNEL_H_

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "base/tril.h"
#include "base/triu.h"
#include "native/cuda/ops/triangular/kernel.cuh"

namespace infini::ops::triangular_detail {

inline std::size_t CheckedElementCount(const Tensor& tensor) {
  std::size_t count = 1;
  for (const auto extent : tensor.shape()) {
    if (extent >
        static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
      throw std::invalid_argument(
          "triangular shape exceeds signed indexing range");
    }
    if (extent && count > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::invalid_argument("triangular element count overflows");
    }
    count *= extent;
  }
  return count;
}

inline std::size_t MaxOffset(const Tensor& tensor) {
  std::size_t offset = 0;
  for (std::size_t dim = 0; dim < tensor.ndim(); ++dim) {
    const auto stride = tensor.strides()[dim];
    if (stride < 0) {
      throw std::invalid_argument(
          "triangular negative strides are unsupported");
    }
    const std::size_t extent = tensor.shape()[dim];
    const std::size_t positive_stride = static_cast<std::size_t>(stride);
    if (extent > 1 &&
        positive_stride > (static_cast<std::size_t>(
                               std::numeric_limits<std::int64_t>::max()) -
                           offset) /
                              (extent - 1)) {
      throw std::invalid_argument("triangular stride span overflows");
    }
    offset += extent ? (extent - 1) * positive_stride : 0;
  }
  return offset;
}

inline bool SameView(const Tensor& input, const Tensor& output) {
  return input.data() == output.data() && input.shape() == output.shape() &&
         input.strides() == output.strides() &&
         input.dtype() == output.dtype() && input.device() == output.device();
}

inline std::pair<std::uintptr_t, std::uintptr_t> ByteSpan(
    const Tensor& tensor, std::size_t elements) {
  if (!elements) return {0, 0};
  const auto maximum = MaxOffset(tensor);
  const auto bytes_per_element = tensor.element_size();
  if (maximum >= std::numeric_limits<std::size_t>::max() / bytes_per_element) {
    throw std::invalid_argument("triangular byte span overflows");
  }
  const auto bytes = (maximum + 1) * bytes_per_element;
  const auto begin = reinterpret_cast<std::uintptr_t>(tensor.data());
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    throw std::invalid_argument("triangular address span overflows");
  }
  return {begin, begin + bytes};
}

inline void ValidateOutputNonOverlapping(const Tensor& output) {
  std::array<std::pair<std::size_t, std::size_t>, kMaxRank> dimensions{};
  std::size_t count = 0;
  for (std::size_t dim = 0; dim < output.ndim(); ++dim) {
    if (output.shape()[dim] > 1) {
      dimensions[count++] = {static_cast<std::size_t>(output.strides()[dim]),
                             output.shape()[dim]};
    }
  }
  std::sort(dimensions.begin(), dimensions.begin() + count);
  std::size_t required = 1;
  for (std::size_t dim = 0; dim < count; ++dim) {
    const auto [stride, extent] = dimensions[dim];
    if (stride < required ||
        stride > std::numeric_limits<std::size_t>::max() / extent) {
      throw std::invalid_argument("triangular output has overlapping strides");
    }
    required = stride * extent;
  }
}

template <typename Key, bool kUpper>
Metadata Validate(const Tensor& input, std::int64_t diagonal,
                  const Tensor& output) {
  static_assert(std::is_same_v<Key, Triu> || std::is_same_v<Key, Tril>);
  if (input.ndim() < 2 || input.ndim() > kMaxRank ||
      input.shape() != output.shape() || input.dtype() != output.dtype() ||
      input.device() != output.device()) {
    throw std::invalid_argument(
        "triangular requires matching rank-2 through rank-8 tensors");
  }
  const auto element_size = input.element_size();
  if (element_size != 1 && element_size != 2 && element_size != 4 &&
      element_size != 8) {
    throw std::invalid_argument("triangular unsupported element size");
  }
  const auto elements = CheckedElementCount(input);
  if (elements && (!input.data() || !output.data())) {
    throw std::invalid_argument("triangular nonempty tensors require storage");
  }
  MaxOffset(input);
  MaxOffset(output);
  ValidateOutputNonOverlapping(output);
  const auto input_span = ByteSpan(input, elements);
  const auto output_span = ByteSpan(output, elements);
  if (input_span.first < output_span.second &&
      output_span.first < input_span.second && !SameView(input, output)) {
    throw std::invalid_argument(
        "triangular input and output partially overlap");
  }
  Metadata meta{};
  meta.rank = static_cast<int>(input.ndim());
  meta.rows = input.shape()[input.ndim() - 2];
  meta.columns = input.shape()[input.ndim() - 1];
  meta.elements = elements;
  meta.diagonal = diagonal;
  for (int dim = 0; dim < meta.rank; ++dim) {
    meta.shape[dim] = input.shape()[dim];
    meta.input_strides[dim] = input.strides()[dim];
    meta.output_strides[dim] = output.strides()[dim];
  }
  return meta;
}

template <typename T, bool kUpper>
void Launch(const Tensor& input, Tensor output, Metadata meta,
            cudaStream_t stream) {
  constexpr unsigned kThreads = 256;
  const std::size_t blocks = 1 + (meta.elements - 1) / kThreads;
  const unsigned grid =
      static_cast<unsigned>(std::min<std::size_t>(blocks, 65535));
  TriangularKernel<T, kUpper>
      <<<grid, kThreads, 0, stream>>>(static_cast<const T*>(input.data()),
                                      static_cast<T*>(output.data()), meta);
  const auto status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

template <typename Backend, typename Key, bool kUpper>
class CudaTriangular : public Key {
 public:
  CudaTriangular(const Tensor input, std::int64_t diagonal, Tensor output)
      : Key(input, diagonal, output),
        metadata_{Validate<Key, kUpper>(input, diagonal, output)} {}

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, std::int64_t /*diagonal*/,
                  Tensor output) const override {
    if (metadata_.elements == 0) return;
    const auto stream = static_cast<typename Backend::Stream>(
        this->stream_ ? this->stream_ : 0);
    switch (input.element_size()) {
      case 1:
        Launch<std::uint8_t, kUpper>(input, output, metadata_, stream);
        break;
      case 2:
        Launch<std::uint16_t, kUpper>(input, output, metadata_, stream);
        break;
      case 4:
        Launch<std::uint32_t, kUpper>(input, output, metadata_, stream);
        break;
      case 8:
        Launch<std::uint64_t, kUpper>(input, output, metadata_, stream);
        break;
      default:
        throw std::invalid_argument("triangular unsupported element size");
    }
  }

 private:
  Metadata metadata_{};
};

}  // namespace infini::ops::triangular_detail

#endif  // INFINI_OPS_CUDA_TRIANGULAR_KERNEL_H_
