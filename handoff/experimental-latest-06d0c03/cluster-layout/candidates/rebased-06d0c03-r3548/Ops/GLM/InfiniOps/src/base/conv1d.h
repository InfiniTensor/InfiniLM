#ifndef INFINI_OPS_BASE_CONV1D_H_
#define INFINI_OPS_BASE_CONV1D_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "common/op_utils/conv.h"

namespace infini::ops {

class Conv1d : public Operator<Conv1d> {
 public:
  Conv1d(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
         const std::vector<int64_t> stride, const std::string padding,
         const std::vector<int64_t> dilation, const int64_t groups, Tensor out)
      : metadata_{conv_detail::MakeMetadata<1>(
            input, weight, bias, stride,
            conv_detail::ResolvePadding<1>(weight, stride, padding, dilation),
            dilation, groups, out)} {}

  Conv1d(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
         const std::vector<int64_t> stride, const std::vector<int64_t> padding,
         const std::vector<int64_t> dilation, const int64_t groups, Tensor out)
      : metadata_{conv_detail::MakeMetadata<1>(
            input, weight, bias, stride,
            conv_detail::ResolvePadding<1>(weight, stride, padding, dilation),
            dilation, groups, out)} {}

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, const std::vector<int64_t> stride,
                  const std::string padding,
                  const std::vector<int64_t> dilation, const int64_t groups,
                  Tensor out) const {
    auto resolved =
        conv_detail::ResolvePadding<1>(weight, stride, padding, dilation);
    (*this)(input, weight, bias, stride, resolved.left, dilation, groups, out);
  }

  virtual void operator()(const Tensor input, const Tensor weight,
                          std::optional<Tensor> bias,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> dilation,
                          const int64_t groups, Tensor out) const = 0;

 protected:
  conv_detail::Metadata metadata_;
};

}  // namespace infini::ops

#endif
