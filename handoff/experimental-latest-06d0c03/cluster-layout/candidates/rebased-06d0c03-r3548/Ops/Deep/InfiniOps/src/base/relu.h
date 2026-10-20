#ifndef INFINI_OPS_BASE_RELU_H_
#define INFINI_OPS_BASE_RELU_H_

#include <cassert>
#include <cstdint>
#include <utility>

#include "operator.h"

namespace infini::ops {

using ReluDataTypes =
    ConcatType<ConcatType<AllFloatTypes, IntTypes>, List<DataType::kUInt8>>;

class Relu : public Operator<Relu> {
 public:
  Relu(const Tensor input, Tensor out)
      : ndim_{out.ndim()},
        output_size_{out.numel()},
        input_type_{input.dtype()},
        out_type_{out.dtype()},
        input_shape_{input.shape()},
        out_shape_{out.shape()},
        input_strides_{input.strides()},
        out_strides_{out.strides()},
        is_input_contiguous_{input.IsContiguous()},
        is_out_contiguous_{out.IsContiguous()} {
    assert(input_shape_ == out_shape_ &&
           "operator `Relu` requires matching input and output shapes");
    assert(input_type_ == out_type_ &&
           "operator `Relu` requires matching input and output dtypes");
    assert(input.device() == out.device() &&
           "operator `Relu` requires input and output on the same device");
    assert(detail::ListContains(input_type_, ReluDataTypes{}) &&
           "operator `Relu` received an unsupported dtype");
    assert(!out.HasBroadcastDim() &&
           "operator `Relu` output must not have broadcasted dimensions");
  }

  virtual void operator()(const Tensor input, Tensor out) const = 0;

 protected:
  bool NeedsInputCopy(const Tensor input, const Tensor out) const {
    if (output_size_ == 0 ||
        (input.data() == out.data() && input.strides() == out.strides())) {
      return false;
    }

    const auto [input_begin, input_end] = StorageByteRange(input);
    const auto [out_begin, out_end] = StorageByteRange(out);

    return input_begin < out_end && out_begin < input_end;
  }

  static std::pair<std::intptr_t, std::intptr_t> StorageByteRange(
      const Tensor tensor) {
    Tensor::Stride min_offset = 0;
    Tensor::Stride max_offset = 0;

    for (Tensor::Size i = 0; i < tensor.ndim(); ++i) {
      const auto extent =
          static_cast<Tensor::Stride>(tensor.size(i) - 1) * tensor.stride(i);

      if (extent < 0) {
        min_offset += extent;
      } else {
        max_offset += extent;
      }
    }

    const auto address = reinterpret_cast<std::intptr_t>(tensor.data());
    const auto element_size = static_cast<std::intptr_t>(tensor.element_size());

    return {address + min_offset * element_size,
            address + (max_offset + 1) * element_size};
  }

  Tensor::Size ndim_{0};

  Tensor::Size output_size_{0};

  const DataType input_type_;

  const DataType out_type_;

  Tensor::Shape input_shape_;

  Tensor::Shape out_shape_;

  Tensor::Strides input_strides_;

  Tensor::Strides out_strides_;

  bool is_input_contiguous_{false};

  bool is_out_contiguous_{false};
};

}  // namespace infini::ops

#endif
