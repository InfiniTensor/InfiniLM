#ifndef INFINI_OPS_BASE_CUDNN_CONVOLUTION_H_
#define INFINI_OPS_BASE_CUDNN_CONVOLUTION_H_

#include <vector>

#include "operator.h"

namespace infini::ops {

class CudnnConvolution : public Operator<CudnnConvolution> {
 public:
  CudnnConvolution(const Tensor input, const Tensor weight,
                   const std::vector<int64_t> padding,
                   const std::vector<int64_t> stride,
                   const std::vector<int64_t> dilation, const int64_t groups,
                   const bool benchmark, const bool deterministic,
                   const bool allow_tf32, Tensor out)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        weight_shape_{weight.shape()},
        weight_strides_{weight.strides()},
        weight_type_{weight.dtype()},
        out_shape_{out.shape()},
        out_strides_{out.strides()},
        out_type_{out.dtype()},
        padding_{padding},
        stride_{stride},
        dilation_{dilation},
        groups_{groups},
        benchmark_{benchmark},
        deterministic_{deterministic},
        allow_tf32_{allow_tf32},
        device_index_{out.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor weight,
                          const std::vector<int64_t> padding,
                          const std::vector<int64_t> stride,
                          const std::vector<int64_t> dilation,
                          const int64_t groups, const bool benchmark,
                          const bool deterministic, const bool allow_tf32,
                          Tensor out) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_;

  Tensor::Shape out_shape_;

  Tensor::Strides out_strides_;

  DataType out_type_;

  std::vector<int64_t> padding_{};

  std::vector<int64_t> stride_{};

  std::vector<int64_t> dilation_{};

  int64_t groups_{};

  bool benchmark_{};

  bool deterministic_{};

  bool allow_tf32_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
