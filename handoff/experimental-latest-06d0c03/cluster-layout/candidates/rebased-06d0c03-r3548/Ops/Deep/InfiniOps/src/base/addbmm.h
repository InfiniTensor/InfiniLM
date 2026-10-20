#ifndef INFINI_OPS_BASE_ADDBMM_H_
#define INFINI_OPS_BASE_ADDBMM_H_

#include "operator.h"

namespace infini::ops {

class Addbmm : public Operator<Addbmm> {
 public:
  Addbmm(const Tensor input, const Tensor batch1, const Tensor batch2,
         const double beta, const double alpha, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        batch1_shape_{batch1.shape()},
        batch1_strides_{batch1.strides()},
        batch1_type_{batch1.dtype()},
        batch2_shape_{batch2.shape()},
        batch2_strides_{batch2.strides()},
        batch2_type_{batch2.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        beta_{beta},
        alpha_{alpha},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor batch1,
                          const Tensor batch2, const double beta,
                          const double alpha, Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape batch1_shape_;

  Tensor::Strides batch1_strides_;

  DataType batch1_type_;

  Tensor::Shape batch2_shape_;

  Tensor::Strides batch2_strides_;

  DataType batch2_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  double beta_{};

  double alpha_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
