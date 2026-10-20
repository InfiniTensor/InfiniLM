#ifndef INFINI_OPS_BASE_REPLICATION_PAD1D_H_
#define INFINI_OPS_BASE_REPLICATION_PAD1D_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class ReplicationPad1d : public Operator<ReplicationPad1d> {
 public:
  ReplicationPad1d(const Tensor input, const std::vector<int64_t> padding,
                   Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        padding_{padding},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input,
                          const std::vector<int64_t> padding,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> padding_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
