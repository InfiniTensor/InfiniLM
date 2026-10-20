#ifndef INFINI_OPS_BASE_FFT_HFFTN_H_
#define INFINI_OPS_BASE_FFT_HFFTN_H_

#include <optional>
#include <string>
#include <vector>

#include "operator.h"

namespace infini::ops::fft {

class Hfftn : public Operator<Hfftn> {
 public:
  Hfftn(const Tensor input, const std::optional<std::vector<int64_t>> s,
        const std::optional<std::vector<int64_t>> dim,
        const std::optional<std::string> norm, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        s_{s},
        dim_{dim},
        norm_{norm},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::optional<std::vector<int64_t>> s,
                          const std::optional<std::vector<int64_t>> dim,
                          const std::optional<std::string> norm,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::optional<std::vector<int64_t>> s_{};

  std::optional<std::vector<int64_t>> dim_{};

  std::optional<std::string> norm_{};

  int device_index_{0};
};

}  // namespace infini::ops::fft

#endif
