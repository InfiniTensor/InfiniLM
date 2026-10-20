#ifndef INFINI_OPS_CAMBRICON_RMS_NORM_H_
#define INFINI_OPS_CAMBRICON_RMS_NORM_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "base/rms_norm.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

// TODO: Remove forward declaration.
template <typename T, typename Tw>
void RmsNormUnion(void* workspace, int core_per_cluster, int cluster_count,
                  cnrtQueue_t queue, void* y, const void* x, const void* w,
                  const size_t* shape, const ptrdiff_t* y_strides,
                  const ptrdiff_t* x_strides, float eps, int ndim);

template <>
class Operator<RmsNorm, Device::Type::kCambricon> : public RmsNorm {
 public:
  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm{input, weight, eps, out} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster,
                                &cluster_count);
    cnrtMalloc(&default_workspace_, workspace_size_in_bytes());
  }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    auto workspace{workspace_ ? workspace_ : default_workspace_};

    DispatchFunc<
        Device::Type::kCambricon,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {input.dtype(), weight.dtype()},
        [&](auto input_tag, auto weight_tag) {
          using InputT = typename decltype(input_tag)::type;
          using WeightT = typename decltype(weight_tag)::type;

          RmsNormUnion<InputT, WeightT>(
              workspace, core_per_cluster, cluster_count, queue, out.data(),
              input.data(), weight.data(), out_shape_.data(),
              out_strides_.data(), input_strides_.data(), eps, ndim_);
        },
        "CambriconRmsNorm::operator() - output dispatch");
  }

  ~Operator() { cnrtFree(default_workspace_); }

  std::size_t workspace_size_in_bytes() const override {
    return ndim_ * (sizeof(size_t) + 2 * sizeof(ptrdiff_t));
  }

  void* default_workspace_{nullptr};
  int core_per_cluster = 0;
  int cluster_count = 0;
};

}  // namespace infini::ops

#endif
