#ifndef INFINI_OPS_BASE_HISTOGRAM_H_
#define INFINI_OPS_BASE_HISTOGRAM_H_

#include <optional>
#include <vector>

#include "operator.h"

namespace infini::ops {

class Histogram : public Operator<Histogram> {
 public:
  Histogram(const Tensor input, const Tensor bins,
            const std::optional<Tensor> weight, const bool density, Tensor hist,
            Tensor bin_edges)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        bins_shape_{bins.shape()},
        bins_strides_{bins.strides()},
        bins_type_{bins.dtype()},
        hist_shape_{hist.shape()},
        hist_strides_{hist.strides()},
        hist_type_{hist.dtype()},
        bin_edges_shape_{bin_edges.shape()},
        bin_edges_strides_{bin_edges.strides()},
        bin_edges_type_{bin_edges.dtype()},
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        density_{density},
        device_index_{hist.device().index()} {}

  Histogram(const Tensor input, const int64_t bins,
            const std::optional<Tensor> weight,
            const std::optional<std::vector<double>> range, const bool density,
            Tensor hist, Tensor bin_edges)
      : input_shape_{input.shape()},
        input_strides_{input.strides()},
        input_type_{input.dtype()},
        hist_shape_{hist.shape()},
        hist_strides_{hist.strides()},
        hist_type_{hist.dtype()},
        bin_edges_shape_{bin_edges.shape()},
        bin_edges_strides_{bin_edges.strides()},
        bin_edges_type_{bin_edges.dtype()},
        has_weight_{weight.has_value()},
        weight_shape_{weight ? Tensor::Shape{weight->shape()}
                             : Tensor::Shape{}},
        weight_strides_{weight ? Tensor::Strides{weight->strides()}
                               : Tensor::Strides{}},
        weight_type_{weight ? weight->dtype() : DataType::kFloat32},
        density_{density},
        bins_{bins},
        range_{range},
        device_index_{hist.device().index()} {}

  virtual void operator()(const Tensor input, const Tensor bins,
                          const std::optional<Tensor> weight,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;

  virtual void operator()(const Tensor input, const int64_t bins,
                          const std::optional<Tensor> weight,
                          const std::optional<std::vector<double>> range,
                          const bool density, Tensor hist,
                          Tensor bin_edges) const = 0;

 protected:
  Tensor::Shape input_shape_;

  Tensor::Strides input_strides_;

  DataType input_type_;

  Tensor::Shape bins_shape_;

  Tensor::Strides bins_strides_;

  DataType bins_type_;

  Tensor::Shape hist_shape_;

  Tensor::Strides hist_strides_;

  DataType hist_type_;

  Tensor::Shape bin_edges_shape_;

  Tensor::Strides bin_edges_strides_;

  DataType bin_edges_type_;

  bool has_weight_{false};

  Tensor::Shape weight_shape_;

  Tensor::Strides weight_strides_;

  DataType weight_type_{DataType::kFloat32};

  bool density_{};

  int64_t bins_{};

  std::optional<std::vector<double>> range_{};

  int device_index_{0};
};

}  // namespace infini::ops

#endif
