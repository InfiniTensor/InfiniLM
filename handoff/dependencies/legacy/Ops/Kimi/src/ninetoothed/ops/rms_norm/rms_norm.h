#ifndef INFINI_OPS_NINETOOTHED_RMS_NORM_H_
#define INFINI_OPS_NINETOOTHED_RMS_NORM_H_

#include <cassert>
#include <cstdint>
#include <vector>

#include "base/rms_norm.h"
#include "data_type.h"
#include "ninetoothed/tensor.h"
#include "rms_norm/infini_ops_ninetoothed_rms_norm.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia, 9> : public RmsNorm {
 public:
  using RmsNorm::RmsNorm;
  using RmsNorm::operator();

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    assert(input.dtype() == out.dtype() && out.dtype() == weight.dtype() &&
           "operator `RmsNorm` requires all input and output tensors to have "
           "the same dtype");
    assert(input.shape() == out.shape() &&
           "NineToothed `RmsNorm` requires input and output tensors with the "
           "same shape");
    assert(weight.ndim() == 1 && weight.size(-1) == out.size(-1) &&
           "NineToothed `RmsNorm` requires a 1D weight matching the last "
           "dimension");
    assert(
        (out.ndim() == 2 || out.ndim() == 3 || out.ndim() == 4) &&
        "NineToothed `RmsNorm` currently supports rank-2, rank-3, and rank-4 "
        "tensors");

    std::vector<std::uint64_t> weight_sizes;
    std::vector<std::int64_t> weight_strides;
    double eps_value = static_cast<double>(eps);
    std::int64_t num_normalized_elements =
        static_cast<std::int64_t>(out.size(-1));
    std::uint64_t empty_shape[1] = {};
    std::int64_t empty_strides[1] = {};

    weight_sizes.assign(out.shape().begin(), out.shape().end());
    weight_strides.assign(out.ndim(), 0);
    weight_strides.back() =
        weight.strides().empty() ? 1 : weight.strides().back();

    const int dtype_index = ninetoothed::DataTypeIndex(out.dtype());
    assert(
        dtype_index >= 0 &&
        "NineToothed `RmsNorm` supports only float16, bfloat16, and float32");

    ninetoothed::Tensor input_tensor(input);
    ninetoothed::Tensor weight_tensor(const_cast<void*>(weight.data()),
                                      weight_sizes.data(),
                                      weight_strides.data());
    ninetoothed::Tensor eps_tensor(eps_value, empty_shape, empty_strides);
    ninetoothed::Tensor out_tensor(out);
    ninetoothed::Tensor num_normalized_elements_tensor(
        num_normalized_elements, empty_shape, empty_strides);

    auto result = launch_infini_ops_ninetoothed_rms_norm(
        static_cast<NineToothedStream>(stream_), input_tensor, weight_tensor,
        eps_tensor, out_tensor, num_normalized_elements_tensor,
        static_cast<int>(out.ndim()), 1, dtype_index, dtype_index, dtype_index);

    assert(result == 0 && "NineToothed `RmsNorm` launch failed");
  }
};

}  // namespace infini::ops

#endif
