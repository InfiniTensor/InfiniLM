#ifndef INFINI_OPS_BASE_LINSPACE_H_
#define INFINI_OPS_BASE_LINSPACE_H_

#include "operator.h"

namespace infini::ops {

class Linspace : public Operator<Linspace> {
 public:
  Linspace(const Tensor start, const Tensor end, const int64_t steps,
           Tensor out)
      : start_shape_{start.shape()},
        start_strides_{start.strides()},
        start_type_{start.dtype()},
        end_shape_{end.shape()},
        end_strides_{end.strides()},
        end_type_{end.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        steps_{steps},
        device_index_{out.device().index()} {}

  Linspace(const Tensor start, const double end, const int64_t steps,
           Tensor out)
      : start_shape_{start.shape()},
        start_strides_{start.strides()},
        start_type_{start.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        steps_{steps},
        end_{end},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor start, const Tensor end,
                          const int64_t steps, Tensor out) const = 0;

  virtual void operator()(const Tensor start, const double end,
                          const int64_t steps, Tensor out) const = 0;

 protected:
  Tensor::Shape start_shape_;

  Tensor::Strides start_strides_;

  DataType start_type_;

  Tensor::Shape end_shape_;

  Tensor::Strides end_strides_;

  DataType end_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  int64_t steps_{};

  double end_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
