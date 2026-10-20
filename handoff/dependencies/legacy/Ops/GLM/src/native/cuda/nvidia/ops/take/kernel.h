#ifndef INFINI_OPS_NVIDIA_TAKE_KERNEL_H_
#define INFINI_OPS_NVIDIA_TAKE_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "base/take.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace nvidia_take_detail {
void Launch(const void* input, const std::int64_t* index, void* out,
            std::size_t input_count, std::size_t output_count,
            std::size_t element_size, cudaStream_t stream);
}  // namespace nvidia_take_detail

template <>
class Operator<Take, Device::Type::kNvidia, 0> : public Take {
 public:
  static bool supports(const Tensor input, const Tensor index, Tensor out) {
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != index.device() || input.device() != out.device() ||
        !CheckedContiguous(input) || !CheckedContiguous(index) ||
        !CheckedContiguous(out) ||
        index.dtype() != DataType::kInt64 || input.dtype() != out.dtype() ||
        index.shape() != out.shape()) {
      return false;
    }
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess ||
        current_device != input.device().index()) return false;
    switch (input.dtype()) {
      case DataType::kInt8: case DataType::kUInt8:
      case DataType::kInt16: case DataType::kUInt16:
      case DataType::kFloat16: case DataType::kBFloat16:
      case DataType::kInt32: case DataType::kUInt32:
      case DataType::kFloat32:
      case DataType::kInt64: case DataType::kUInt64:
      case DataType::kFloat64:
        break;
      default:
        return false;
    }
    std::size_t input_count, output_count;
    if (!CheckedNumel(input, &input_count) ||
        !CheckedNumel(index, &output_count) ||
        input_count > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
        (output_count && !input_count)) {
      return false;
    }
    const auto width = input.element_size();
    if (!width || input_count > std::numeric_limits<std::size_t>::max() / width ||
        output_count > std::numeric_limits<std::size_t>::max() / width ||
        output_count > std::numeric_limits<std::size_t>::max() / sizeof(std::int64_t)) {
      return false;
    }
    if ((input_count && !input.data()) ||
        (output_count && (!index.data() || !out.data())) ||
        (input_count && reinterpret_cast<std::uintptr_t>(input.data()) % width) ||
        (output_count && (reinterpret_cast<std::uintptr_t>(out.data()) % width ||
                          reinterpret_cast<std::uintptr_t>(index.data()) % alignof(std::int64_t)))) {
      return false;
    }
    return !Overlaps(input.data(), input_count * width, index.data(),
                     output_count * sizeof(std::int64_t)) &&
           !Overlaps(input.data(), input_count * width, out.data(),
                     output_count * width) &&
           !Overlaps(index.data(), output_count * sizeof(std::int64_t),
                     out.data(), output_count * width);
  }

  Operator(const Tensor input, const Tensor index, Tensor out)
      : Take(input, index, out) {
    Validate(input, index, out);
  }

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const Tensor index, Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        input.dtype() != input_type_ || index.shape() != index_shape_ ||
        index.strides() != index_strides_ || index.dtype() != index_type_ ||
        out.shape() != out_shape_ || out.strides() != out_strides_ ||
        out.dtype() != out_type_ || out.device().index() != device_index_) {
      throw std::invalid_argument("NVIDIA Take call metadata changed after Make");
    }
    Validate(input, index, out);
    const auto output_count = out.numel();
    if (!output_count) return;
    nvidia_take_detail::Launch(input.data(),
                               static_cast<const std::int64_t*>(index.data()),
                               out.data(), input.numel(), output_count,
                               input.element_size(),
                               static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static bool CheckedContiguous(const Tensor tensor) {
    const auto& shape = tensor.shape();
    const auto& strides = tensor.strides();
    if (shape.size() != strides.size()) return false;
    std::size_t expected = 1;
    for (std::size_t i = shape.size(); i > 0; --i) {
      if (expected > static_cast<std::size_t>(
                         std::numeric_limits<Tensor::Stride>::max()) ||
          strides[i - 1] != static_cast<Tensor::Stride>(expected)) return false;
      if (shape[i - 1] &&
          expected > std::numeric_limits<std::size_t>::max() / shape[i - 1])
        return false;
      expected *= shape[i - 1];
    }
    return true;
  }

  static bool CheckedNumel(const Tensor tensor, std::size_t* result) {
    *result = 1;
    for (const auto dim : tensor.shape()) {
      if (dim && *result > std::numeric_limits<std::size_t>::max() / dim) return false;
      *result *= dim;
    }
    return true;
  }

  static bool Overlaps(const void* a, std::size_t a_bytes,
                       const void* b, std::size_t b_bytes) {
    if (!a_bytes || !b_bytes) return false;
    const auto start_a = reinterpret_cast<std::uintptr_t>(a);
    const auto start_b = reinterpret_cast<std::uintptr_t>(b);
    const auto max = std::numeric_limits<std::uintptr_t>::max();
    if (a_bytes > max - start_a || b_bytes > max - start_b) return true;
    return start_a < start_b + b_bytes && start_b < start_a + a_bytes;
  }

  static void Validate(const Tensor input, const Tensor index, Tensor out) {
    if (!supports(input, index, out))
      throw std::invalid_argument("unsupported NVIDIA Take metadata, device or alias");
  }
};

}  // namespace infini::ops

#endif  // INFINI_OPS_NVIDIA_TAKE_KERNEL_H_
