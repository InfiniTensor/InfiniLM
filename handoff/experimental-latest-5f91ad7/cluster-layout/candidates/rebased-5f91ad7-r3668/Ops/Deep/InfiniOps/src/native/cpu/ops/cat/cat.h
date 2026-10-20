#ifndef INFINI_OPS_CPU_CAT_CAT_H_
#define INFINI_OPS_CPU_CAT_CAT_H_

#include <cstring>
#include <vector>

#include "base/cat.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kCpu> : public Cat {
 public:
  Operator(const std::vector<Tensor> tensors, const int64_t dim, Tensor out)
      : Cat{tensors, dim, out} {}

  void operator()(const std::vector<Tensor> tensors, const int64_t /*dim*/,
                  Tensor out) const override {
    auto elem_size = kDataTypeToSize.at(out.dtype());
    auto* out_ptr = static_cast<char*>(out.data());
    Tensor::Size cat_offset = 0;

    for (const auto& tensor : tensors) {
      auto* input_ptr = static_cast<const char*>(tensor.data());

      for (Tensor::Size flat_index = 0; flat_index < tensor.numel();
           ++flat_index) {
        auto remaining = flat_index;
        Tensor::Stride input_offset = 0;
        Tensor::Stride output_offset = 0;

        for (Tensor::Size axis = tensor.ndim(); axis-- > 0;) {
          auto coordinate = remaining % tensor.size(axis);
          remaining /= tensor.size(axis);
          input_offset += coordinate * tensor.stride(axis);
          output_offset +=
              (coordinate +
               (axis == static_cast<Tensor::Size>(dim_) ? cat_offset : 0)) *
              out.stride(axis);
        }

        std::memcpy(out_ptr + output_offset * elem_size,
                    input_ptr + input_offset * elem_size, elem_size);
      }
      cat_offset += tensor.size(dim_);
    }
  }
};

}  // namespace infini::ops

#endif
