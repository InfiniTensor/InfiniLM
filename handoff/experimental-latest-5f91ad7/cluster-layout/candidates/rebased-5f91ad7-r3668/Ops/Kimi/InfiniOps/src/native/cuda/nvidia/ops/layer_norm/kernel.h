#ifndef INFINI_OPS_NVIDIA_LAYER_NORM_KERNEL_H_
#define INFINI_OPS_NVIDIA_LAYER_NORM_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>

#include "base/layer_norm.h"
#include "native/cuda/nvidia/ops/binary_view_capture.h"
#include "native/cuda/nvidia/runtime_.h"

namespace infini::ops {
namespace nvidia_layer_norm_detail {
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
            DataType dtype, Layout layout, std::size_t rows, float eps,
            cudaStream_t stream);
}  // namespace nvidia_layer_norm_detail

template <>
class Operator<LayerNorm, Device::Type::kNvidia, 0> : public LayerNorm {
 public:
  static bool supports(const Tensor input, const Tensor weight,
                       const std::optional<Tensor> bias, float, Tensor out) {
    using namespace nvidia_binary_view;
    const auto rank = input.shape().size();
    if (input.device().type() != Device::Type::kNvidia ||
        input.device() != weight.device() || input.device() != out.device() ||
        (bias && input.device() != bias->device()) || rank < 1 || rank > 8 ||
        rank != input.strides().size() || out.shape() != input.shape() ||
        out.strides().size() != rank || weight.shape().size() != 1 ||
        weight.strides().size() != 1 ||
        (bias && (bias->shape().size() != 1 || bias->strides().size() != 1)) ||
        input.shape()[rank - 1] == 0 ||
        weight.shape()[0] != input.shape()[rank - 1] ||
        (bias && bias->shape()[0] != input.shape()[rank - 1]) ||
        input.dtype() != weight.dtype() || input.dtype() != out.dtype() ||
        (bias && bias->dtype() != input.dtype()) ||
        (input.dtype() != DataType::kFloat16 &&
         input.dtype() != DataType::kBFloat16 &&
         input.dtype() != DataType::kFloat32) ||
        !NonOverlappingOutput(out))
      return false;

    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        current != input.device().index())
      return false;
    if (input.dtype() == DataType::kBFloat16) {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, current) != cudaSuccess ||
          prop.major < 8)
        return false;
    }
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
        weight.strides()[0] < 0 || (bias && bias->strides()[0] < 0))
      return false;
    std::size_t input_bytes, weight_bytes, out_bytes, bias_bytes = 0;
    if (!SpanBytes(input, &input_bytes) || !SpanBytes(weight, &weight_bytes) ||
        !SpanBytes(out, &out_bytes) || (bias && !SpanBytes(*bias, &bias_bytes)))
      return false;
    const auto width = input.element_size();
    if ((input_bytes &&
         reinterpret_cast<std::uintptr_t>(input.data()) % width) ||
        (weight_bytes &&
         reinterpret_cast<std::uintptr_t>(weight.data()) % width) ||
        (out_bytes && reinterpret_cast<std::uintptr_t>(out.data()) % width) ||
        (bias_bytes && reinterpret_cast<std::uintptr_t>(bias->data()) % width))
      return false;
    const bool exact_inplace = input.data() == out.data() &&
                               input.strides() == out.strides() &&
                               NonOverlappingOutput(input);
    return !Overlaps(out, out_bytes, weight, weight_bytes) &&
           (!bias || !Overlaps(out, out_bytes, *bias, bias_bytes)) &&
           (!Overlaps(out, out_bytes, input, input_bytes) || exact_inplace);
  }

  Operator(const Tensor input, const Tensor weight,
           const std::optional<Tensor> bias, float eps, Tensor out)
      : LayerNorm(checked(input, weight, bias, eps, out), weight, bias, eps,
                  out) {
    using namespace nvidia_binary_view;
    SpanBytes(input, &input_bytes_);
    SpanBytes(weight, &weight_bytes_);
    SpanBytes(out, &out_bytes_);
    if (bias) SpanBytes(*bias, &bias_bytes_);
    // An identical input layout inherits the validated output non-overlap.
    exact_inplace_layout_ = input_meta_.strides() == out_meta_.strides();
    layout_.rank = static_cast<unsigned>(input.ndim());
    layout_.dim = input.size(-1);
    layout_.weight_stride = static_cast<std::size_t>(weight.stride(0));
    layout_.bias_stride = bias ? static_cast<std::size_t>(bias->stride(0)) : 0;
    for (std::size_t i = 0; i < input.ndim(); ++i) {
      layout_.shape[i] = input.shape()[i];
      layout_.input_strides[i] = static_cast<std::size_t>(input.strides()[i]);
      layout_.out_strides[i] = static_cast<std::size_t>(out.strides()[i]);
      if (i + 1 < input.ndim()) rows_ *= input.shape()[i];
    }
  }

  // Replay submits a kernel with by-value layout and no device allocation.
  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const Tensor weight,
                  const std::optional<Tensor> bias, float eps,
                  Tensor out) const override {
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(weight, weight_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) ||
        bias.has_value() != bias_meta_.has_value() ||
        (bias && !std::equal_to<Tensor>{}(*bias, *bias_meta_)) ||
        bits(eps) != bits(eps_))
      throw std::invalid_argument(
          "NVIDIA LayerNorm call metadata or alias changed after Make");

    using namespace nvidia_binary_view;
    const auto bound = [](const Tensor tensor, std::size_t bytes) {
      return !bytes ||
             (tensor.data() && reinterpret_cast<std::uintptr_t>(tensor.data()) %
                                       tensor.element_size() ==
                                   0);
    };
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != device_.index() ||
        !bound(input, input_bytes_) || !bound(weight, weight_bytes_) ||
        !bound(out, out_bytes_) || (bias && !bound(*bias, bias_bytes_)) ||
        Overlaps(out, out_bytes_, weight, weight_bytes_) ||
        (bias && Overlaps(out, out_bytes_, *bias, bias_bytes_)) ||
        (Overlaps(out, out_bytes_, input, input_bytes_) &&
         !(input.data() == out.data() && exact_inplace_layout_)))
      throw std::invalid_argument(
          "NVIDIA LayerNorm current device, pointer or alias changed after "
          "Make");
    if (!rows_) return;
    nvidia_layer_norm_detail::Launch(
        input.data(), weight.data(), bias ? bias->data() : nullptr, out.data(),
        input_meta_.dtype(), layout_, rows_, eps_,
        static_cast<cudaStream_t>(stream_ ? stream_ : 0));
  }

 private:
  nvidia_layer_norm_detail::Layout layout_{};
  std::size_t rows_ = 1;
  std::size_t input_bytes_ = 0, weight_bytes_ = 0, out_bytes_ = 0,
              bias_bytes_ = 0;
  bool exact_inplace_layout_ = false;

  static std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
  }
  static Tensor checked(const Tensor input, const Tensor weight,
                        const std::optional<Tensor> bias, float eps,
                        Tensor out) {
    if (!supports(input, weight, bias, eps, out))
      throw std::invalid_argument(
          "unsupported NVIDIA LayerNorm metadata, device or alias");
    return input;
  }
};
}  // namespace infini::ops

#endif
