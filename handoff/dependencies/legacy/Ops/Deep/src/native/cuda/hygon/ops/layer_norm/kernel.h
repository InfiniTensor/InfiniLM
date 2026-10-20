#ifndef INFINI_OPS_HYGON_LAYER_NORM_KERNEL_H_
#define INFINI_OPS_HYGON_LAYER_NORM_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

#include "base/layer_norm.h"
#include "native/cuda/hygon/ops/binary_view_capture.h"
#include "native/cuda/hygon/runtime_.h"

namespace infini::ops {
namespace hygon_layer_norm_detail {
struct Layout {
  std::size_t shape[8];
  std::size_t input_strides[8];
  std::size_t out_strides[8];
  std::size_t weight_stride;
  std::size_t bias_stride;
  std::size_t dim;
  unsigned rank;
};

void Launch(const void* input, const void* weight, const void* bias, void* out,
            DataType dtype, Layout layout, std::size_t rows,
            float eps, cudaStream_t stream);
}  // namespace hygon_layer_norm_detail

template <>
class Operator<LayerNorm, Device::Type::kHygon, 0> : public LayerNorm {
 public:
  static bool supports(const Tensor input, const Tensor weight,
                       const std::optional<Tensor> bias, float eps, Tensor out) {
    using namespace hygon_binary_view;
    const auto rank = input.shape().size();
    if (!std::isfinite(eps) || eps <= 0) return false;
    if (input.device().type() != Device::Type::kHygon ||
        input.device() != weight.device() || input.device() != out.device() ||
        (bias && input.device() != bias->device()) ||
        rank < 1 || rank > 8 || rank != input.strides().size() ||
        out.shape() != input.shape() || out.strides().size() != rank ||
        weight.shape().size() != 1 || weight.strides().size() != 1 ||
        (bias && (bias->shape().size() != 1 || bias->strides().size() != 1)) ||
        input.shape()[rank - 1] == 0 ||
        weight.shape()[0] != input.shape()[rank - 1] ||
        (bias && bias->shape()[0] != input.shape()[rank - 1]) ||
        input.dtype() != weight.dtype() || input.dtype() != out.dtype() ||
        (bias && bias->dtype() != input.dtype()) ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) ||
        (input.shape()[rank - 1] > 1 &&
         (input.strides()[rank - 1] != 1 || out.strides()[rank - 1] != 1)) ||
        !NonOverlappingOutput(out)) return false;

    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        current != input.device().index()) return false;
    std::size_t rows = 1;
    for (std::size_t i = 0; i + 1 < rank; ++i) {
      if (input.shape()[i] &&
          rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) /
                     input.shape()[i])
        return false;
      rows *= input.shape()[i];
    }
    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input.shape()[rank - 1] >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        weight.strides()[0] < 0 ||
        (bias && bias->strides()[0] < 0)) return false;
    std::size_t input_bytes, weight_bytes, out_bytes, bias_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(weight, &weight_bytes) ||
        !SpanBytes(out, &out_bytes) ||
        (bias && !SpanBytes(*bias, &bias_bytes))) return false;
    const auto width = input.element_size();
    if ((input_bytes && reinterpret_cast<std::uintptr_t>(input.data()) % width) ||
        (weight_bytes && reinterpret_cast<std::uintptr_t>(weight.data()) % width) ||
        (out_bytes && reinterpret_cast<std::uintptr_t>(out.data()) % width) ||
        (bias_bytes && reinterpret_cast<std::uintptr_t>(bias->data()) % width))
      return false;
    const bool exact_inplace = input.data() == out.data() &&
        input.strides() == out.strides() && NonOverlappingOutput(input);
    return !Overlaps(out, out_bytes, weight, weight_bytes) &&
        (!bias || !Overlaps(out, out_bytes, *bias, bias_bytes)) &&
        (!Overlaps(out, out_bytes, input, input_bytes) || exact_inplace);
  }

  Operator(const Tensor input, const Tensor weight,
           const std::optional<Tensor> bias, float eps, Tensor out)
      : LayerNorm(checked(input, weight, bias, eps, out), weight, bias, eps, out) {}

  // Keep graph qualification closed until the independent CUDA/RT contract
  // has run on the target architecture. Direct capture is tested separately.
  bool capture_safe() const override { return false; }

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<Tensor> bias, float eps,
                  Tensor out) const override {
    if (input.shape() != input_shape_ || input.strides() != input_strides_ ||
        weight.shape() != weight_shape_ || weight.strides() != weight_strides_ ||
        out.shape() != out_shape_ || out.strides() != out_strides_ ||
        input.dtype() != dtype_ || weight.dtype() != dtype_ ||
        out.dtype() != dtype_ || input.device() != device_ ||
        bias.has_value() != has_bias_ ||
        (bias && (bias->shape() != bias_shape_ ||
                  bias->strides() != bias_strides_ || bias->dtype() != dtype_)) ||
        bits(eps) != bits(eps_) || !supports(input, weight, bias, eps, out))
      throw std::invalid_argument("HYGON LayerNorm call metadata or alias changed after Make");

    std::size_t rows = 1;
    hygon_layer_norm_detail::Layout layout{};
    layout.rank = static_cast<unsigned>(input.ndim());
    layout.dim = input.size(-1);
    layout.weight_stride = static_cast<std::size_t>(weight.stride(0));
    layout.bias_stride = bias ? static_cast<std::size_t>(bias->stride(0)) : 0;
    for (std::size_t i = 0; i < input.ndim(); ++i) {
      layout.shape[i] = input.shape()[i];
      layout.input_strides[i] = static_cast<std::size_t>(input.strides()[i]);
      layout.out_strides[i] = static_cast<std::size_t>(out.strides()[i]);
      if (i + 1 < input.ndim()) rows *= input.shape()[i];
    }
    if (!rows) return;
    hygon_layer_norm_detail::Launch(
        input.data(), weight.data(), bias ? bias->data() : nullptr,
        out.data(), dtype_, layout, rows, eps,
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  static std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }
  static Tensor checked(const Tensor input, const Tensor weight,
                        const std::optional<Tensor> bias, float eps, Tensor out) {
    if (!supports(input, weight, bias, eps, out))
      throw std::invalid_argument("unsupported HYGON LayerNorm metadata, device or alias");
    return input;
  }
};
}  // namespace infini::ops

#endif
