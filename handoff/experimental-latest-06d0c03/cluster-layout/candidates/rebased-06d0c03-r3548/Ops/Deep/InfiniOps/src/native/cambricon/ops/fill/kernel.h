#ifndef INFINI_OPS_CAMBRICON_FILL_KERNEL_H_
#define INFINI_OPS_CAMBRICON_FILL_KERNEL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "base/fill.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

using CambriconFillTypes =
    ConcatType<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
               AllIntTypes>;

template <typename T>
void FillUnion(void* workspace, cnrtQueue_t queue, int core_per_cluster,
               int cluster_count, const T* device_value, double host_value,
               T* out, const std::size_t* out_shape,
               const ptrdiff_t* out_strides, std::size_t output_size, int ndim,
               bool out_contiguous);

void FillRaw64Union(void* workspace, cnrtQueue_t queue, int core_per_cluster,
                    int cluster_count, const void* device_value, void* out,
                    const std::size_t* out_shape, const ptrdiff_t* out_strides,
                    std::size_t output_size, int ndim, bool out_contiguous);

template <>
class Operator<Fill, Device::Type::kCambricon> : public Fill {
 public:
  Operator(const Tensor input, const double value, Tensor out)
      : Fill{input, value, out} {
    Initialize(input, out);
  }

  Operator(const Tensor input, const Tensor value, Tensor out)
      : Fill{input, value, out} {
    assert(value.numel() == 1 &&
           "`CambriconFill` requires a scalar Tensor value.");
    assert(value.dtype() == out.dtype() &&
           "`CambriconFill` requires Tensor value and output to have the same "
           "dtype.");
    assert(value.device() == out.device() &&
           "`CambriconFill` requires Tensor value and output on the same "
           "device.");
    Initialize(input, out);
  }

  void operator()(const Tensor input, const double value,
                  Tensor out) const override {
    (void)input;
    Run(nullptr, value, out);
  }

  void operator()(const Tensor input, const Tensor value,
                  Tensor out) const override {
    (void)input;
    Run(value.data(), 0.0, out);
  }

  std::size_t workspace_size_in_bytes() const override {
    return sizeof(std::uint64_t) +
           ndim_ * (sizeof(std::size_t) + sizeof(ptrdiff_t));
  }

 private:
  void Initialize(const Tensor input, const Tensor out) {
    assert(input.shape() == out.shape() &&
           "`CambriconFill` requires input and output to have the same "
           "shape.");
    assert(input.dtype() == out.dtype() &&
           "`CambriconFill` requires input and output to have the same "
           "dtype.");
    assert(input.device() == out.device() &&
           "`CambriconFill` requires input and output on the same device.");
    assert(!out.HasBroadcastDim() &&
           "`CambriconFill` output must not have broadcast dimensions.");

    output_size_ = out.numel();
    ndim_ = out.ndim();
    is_out_contiguous_ = out.IsContiguous();
    cnrt_utils::GetLaunchConfig(out.device(), &core_per_cluster_,
                                &cluster_count_);
    default_workspace_ =
        cnrt_utils::AllocateDeviceBuffer(workspace_size_in_bytes());
  }

  void Run(const void* device_value, double host_value, Tensor out) const {
    if (output_size_ == 0) {
      return;
    }

    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    void* workspace = workspace_ ? workspace_ : default_workspace_.get();

    DispatchFunc<Device::Type::kCambricon, CambriconFillTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          const T* effective_device_value = static_cast<const T*>(device_value);
          if constexpr (std::is_same_v<T, std::int64_t> ||
                        std::is_same_v<T, std::uint64_t>) {
            if (effective_device_value == nullptr) {
              const T converted_host_value = static_cast<T>(host_value);
              CNRT_CHECK(cnrtMemcpy(workspace,
                                    const_cast<T*>(&converted_host_value),
                                    sizeof(T), cnrtMemcpyHostToDev));
              effective_device_value = static_cast<const T*>(workspace);
            }
          }
          if constexpr (std::is_same_v<T, std::int64_t> ||
                        std::is_same_v<T, std::uint64_t>) {
            FillRaw64Union(workspace, queue, core_per_cluster_, cluster_count_,
                           effective_device_value, out.data(),
                           out_shape_.data(), out_strides_.data(), output_size_,
                           static_cast<int>(ndim_), is_out_contiguous_);
          } else {
            FillUnion<T>(workspace, queue, core_per_cluster_, cluster_count_,
                         effective_device_value, host_value,
                         static_cast<T*>(out.data()), out_shape_.data(),
                         out_strides_.data(), output_size_,
                         static_cast<int>(ndim_), is_out_contiguous_);
          }
        },
        "CambriconFill::operator()");
  }

  std::size_t output_size_{0};

  std::size_t ndim_{0};

  bool is_out_contiguous_{false};

  cnrt_utils::DeviceBuffer default_workspace_{};

  int core_per_cluster_{0};

  int cluster_count_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_FILL_KERNEL_H_
