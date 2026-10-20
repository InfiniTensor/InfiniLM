#ifndef INFINI_OPS_CAMBRICON_MUL_KERNEL_H_
#define INFINI_OPS_CAMBRICON_MUL_KERNEL_H_

#include <cstddef>

#include "base/mul.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void MulUnion(void* workspace, cnrtQueue_t queue, int core_per_cluster,
              int cluster_count, const T* input, const T* other, T* out,
              const std::size_t* out_shape, const ptrdiff_t* input_strides,
              const ptrdiff_t* other_strides, const ptrdiff_t* out_strides,
              std::size_t output_size, int ndim, bool fast_path,
              bool out_contiguous);

template <>
class Operator<Mul, Device::Type::kCambricon> : public Mul {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Mul{input, other, out},
        default_workspace_{
            cnrt_utils::AllocateDeviceBuffer(workspace_size_in_bytes())} {
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster_,
                                &cluster_count_);
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    const bool fast_path = is_input_contiguous_ && is_other_contiguous_ &&
                           is_out_contiguous_ && input.shape() == out.shape() &&
                           other.shape() == out.shape();
    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    void* workspace = workspace_ ? workspace_ : default_workspace_.get();

    using SupportedTypes =
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32,
             DataType::kInt16, DataType::kInt32, DataType::kInt64,
             DataType::kUInt16, DataType::kUInt32, DataType::kUInt64>;
    DispatchFunc<Device::Type::kCambricon, SupportedTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          MulUnion<T>(workspace, queue, core_per_cluster_, cluster_count_,
                      static_cast<const T*>(input.data()),
                      static_cast<const T*>(other.data()),
                      static_cast<T*>(out.data()), out_shape_.data(),
                      input_strides_.data(), other_strides_.data(),
                      out_strides_.data(), output_size_,
                      static_cast<int>(ndim_), fast_path, is_out_contiguous_);
        },
        "CambriconMul::operator()");
  }

  std::size_t workspace_size_in_bytes() const override {
    return ndim_ * (sizeof(std::size_t) + 3 * sizeof(ptrdiff_t));
  }

 private:
  cnrt_utils::DeviceBuffer default_workspace_{};

  int core_per_cluster_{0};

  int cluster_count_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_MUL_KERNEL_H_
