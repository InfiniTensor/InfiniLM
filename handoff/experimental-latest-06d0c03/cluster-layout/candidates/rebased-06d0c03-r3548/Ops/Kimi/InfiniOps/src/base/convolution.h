#ifndef INFINI_OPS_BASE_CONVOLUTION_H_
#define INFINI_OPS_BASE_CONVOLUTION_H_

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <optional>
#include <vector>

#include "common/op_utils/conv.h"

namespace infini::ops {

class Convolution : public Operator<Convolution> {
 public:
  Convolution(const Tensor input, const Tensor weight,
              std::optional<Tensor> bias, const std::vector<int64_t> stride,
              const std::vector<int64_t> padding,
              const std::vector<int64_t> dilation, const bool transposed,
              const std::vector<int64_t> output_padding, const int64_t groups,
              Tensor out)
      : metadata_{MakeMetadata(input, weight, bias, stride, padding, dilation,
                               transposed, output_padding, groups, out)} {}

  virtual void operator()(const Tensor input, const Tensor weight,
                          std::optional<Tensor> bias,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> dilation,
                          const bool transposed,
                          const std::vector<int64_t> output_padding,
                          const int64_t groups, Tensor out) const = 0;

 protected:
  conv_detail::Metadata metadata_;

 private:
  static conv_detail::Metadata MakeMetadata(
      const Tensor input, const Tensor weight, std::optional<Tensor> bias,
      const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
      const std::vector<int64_t>& dilation, const bool transposed,
      const std::vector<int64_t>& output_padding, const int64_t groups,
      Tensor out) {
    assert((input.ndim() >= 3 && input.ndim() <= 5) &&
           "operator `Convolution` currently supports only 1D, 2D, and 3D "
           "inputs");
    assert(!transposed &&
           "operator `Convolution` does not currently support transposed "
           "convolution");
    assert(output_padding.size() + 2 == input.ndim() &&
           "operator `Convolution` `output_padding` has the wrong length");
    assert(std::all_of(output_padding.begin(), output_padding.end(),
                       [](int64_t value) { return value == 0; }) &&
           "operator `Convolution` does not currently support nonzero "
           "`output_padding` values");

    switch (input.ndim()) {
      case 3:
        return conv_detail::MakeMetadata<1>(
            input, weight, bias, stride,
            conv_detail::ResolvePadding<1>(weight, stride, padding, dilation),
            dilation, groups, out);
      case 4:
        return conv_detail::MakeMetadata<2>(
            input, weight, bias, stride,
            conv_detail::ResolvePadding<2>(weight, stride, padding, dilation),
            dilation, groups, out);
      case 5:
        return conv_detail::MakeMetadata<3>(
            input, weight, bias, stride,
            conv_detail::ResolvePadding<3>(weight, stride, padding, dilation),
            dilation, groups, out);
      default:
        return {};
    }
  }
};

}  // namespace infini::ops

#endif
