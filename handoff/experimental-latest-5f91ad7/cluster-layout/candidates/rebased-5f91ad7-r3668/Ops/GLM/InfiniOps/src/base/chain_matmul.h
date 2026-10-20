#ifndef INFINI_OPS_BASE_CHAIN_MATMUL_H_
#define INFINI_OPS_BASE_CHAIN_MATMUL_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class ChainMatmul : public Operator<ChainMatmul> {
 public:
  ChainMatmul(const std::vector<Tensor> matrices, Tensor out)
      : out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        matrices_{matrices},
        device_index_{out.device().index()} {}

  virtual void operator()(const std::vector<Tensor> matrices,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<Tensor> matrices_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
