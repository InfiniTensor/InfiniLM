#ifndef INFINI_OPS_CAMBRICON_FUSED_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CAMBRICON_FUSED_ADD_RMS_NORM_KERNEL_H_

#include <optional>

#include "base/fused_add_rms_norm.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void FusedAddRmsNormUnion(int core_per_cluster, int cluster_count,
                          cnrtQueue_t queue, void* input, void* residual,
                          const void* weight, size_t num_tokens, size_t dim,
                          ptrdiff_t input_row_stride,
                          ptrdiff_t residual_row_stride, float epsilon);

template <>
class Operator<FusedAddRmsNorm, Device::Type::kCambricon>
    : public FusedAddRmsNorm {
 public:
  using FusedAddRmsNorm::FusedAddRmsNorm;

  Operator(Tensor input, Tensor residual, const std::optional<Tensor> weight,
           float epsilon)
      : FusedAddRmsNorm{input, residual, weight, epsilon} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster_,
                                &cluster_count_);
  }

  void operator()(Tensor input, Tensor residual,
                  const std::optional<Tensor> weight,
                  float epsilon) const override {
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);

    DispatchFunc<
        Device::Type::kCambricon,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {input.dtype()},
        [&](auto input_tag) {
          using T = typename decltype(input_tag)::type;
          FusedAddRmsNormUnion<T>(
              core_per_cluster_, cluster_count_, queue, input.data(),
              residual.data(), weight.has_value() ? weight->data() : nullptr,
              num_tokens_, dim_, input_strides_[input_strides_.size() - 2],
              residual_strides_[residual_strides_.size() - 2], epsilon);
        },
        "CambriconFusedAddRmsNorm::operator()");
  }

  std::size_t workspace_size_in_bytes() const override { return 0; }

 private:
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif
