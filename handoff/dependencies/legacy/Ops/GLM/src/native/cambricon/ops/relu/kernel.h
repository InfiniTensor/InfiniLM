#ifndef INFINI_OPS_CAMBRICON_RELU_KERNEL_H_
#define INFINI_OPS_CAMBRICON_RELU_KERNEL_H_

#include <cassert>
#include <cstddef>

#include "base/relu.h"
#include "native/cambricon/cnrt_utils.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void ReluUnion(void* workspace, int core_per_cluster, int cluster_count,
               cnrtQueue_t queue, const void* input, void* out,
               const size_t* shape, const ptrdiff_t* input_strides,
               const ptrdiff_t* out_strides, size_t output_size, int ndim,
               bool input_contiguous, bool out_contiguous,
               bool needs_input_copy);

template <>
class Operator<Relu, Device::Type::kCambricon> : public Relu {
 public:
  Operator(const Tensor input, Tensor out)
      : Relu{input, out}, element_size_{out.element_size()} {
    assert(input_type_ != DataType::kFloat64 &&
           "`CambriconRelu` does not support float64 because the Cambricon "
           "device compiler does not support float64 comparisons.");
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster_,
                                &cluster_count_);
    const auto workspace_size = workspace_size_in_bytes();
    if (workspace_size != 0) {
      CNRT_CHECK(cnrtMalloc(&default_workspace_, workspace_size));
    }
  }

  ~Operator() {
    if (default_workspace_) {
      (void)cnrtFree(default_workspace_);
    }
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    auto* workspace = workspace_ ? workspace_ : default_workspace_;
    const auto available_workspace_size =
        workspace_ ? workspace_size_in_bytes_ : workspace_size_in_bytes();
    assert(available_workspace_size >= workspace_size_in_bytes() &&
           "`CambriconRelu` requires a sufficiently large workspace.");
    const bool needs_input_copy = NeedsInputCopy(input, out);

    DispatchFunc<Device::Type::kCambricon,
                 List<DataType::kFloat32, DataType::kFloat16,
                      DataType::kBFloat16, DataType::kInt64, DataType::kInt32,
                      DataType::kInt16, DataType::kInt8, DataType::kUInt8>>(
        {out_type_},
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          ReluUnion<T>(workspace, core_per_cluster_, cluster_count_, queue,
                       input.data(), out.data(), input_shape_.data(),
                       input_strides_.data(), out_strides_.data(), output_size_,
                       static_cast<int>(ndim_), is_input_contiguous_,
                       is_out_contiguous_, needs_input_copy);
        },
        "CambriconRelu::operator() - output dispatch");
  }

  std::size_t workspace_size_in_bytes() const override {
    const auto metadata_size = ndim_ * (sizeof(size_t) + 2 * sizeof(ptrdiff_t));
    return metadata_size + output_size_ * element_size_;
  }

 private:
  std::size_t element_size_{0};
  void* default_workspace_{nullptr};
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif
