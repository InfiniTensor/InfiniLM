#ifndef INFINI_OPS_BASE_INTERNAL_FFT_C2R_H_
#define INFINI_OPS_BASE_INTERNAL_FFT_C2R_H_

#include <vector>

#include "operator.h"

namespace infini::ops::internal::fft {

class C2r : public Operator<C2r> {
 public:
  C2r(const Tensor input, const std::vector<int64_t> dim,
      const int64_t normalization, const int64_t last_dim_size, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        dim_{dim},
        normalization_{normalization},
        last_dim_size_{last_dim_size},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const std::vector<int64_t> dim,
                          const int64_t normalization,
                          const int64_t last_dim_size, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> dim_{};

  int64_t normalization_{};

  int64_t last_dim_size_{};

  int device_index_{0};
};

}  // namespace infini::ops::internal::fft

#endif
