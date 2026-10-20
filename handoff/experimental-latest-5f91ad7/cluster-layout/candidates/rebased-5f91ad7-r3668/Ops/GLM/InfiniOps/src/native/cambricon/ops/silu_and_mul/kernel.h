#ifndef INFINI_OPS_CAMBRICON_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_CAMBRICON_SILU_AND_MUL_KERNEL_H_

#include <cstddef>

#include "base/silu_and_mul.h"
#include "native/cambricon/common.h"
#include "native/cambricon/data_type_.h"

namespace infini::ops {

template <typename T>
void SiluAndMulUnion(void* workspace, int core_per_cluster, int cluster_count,
                     cnrtQueue_t queue, const void* input, void* out,
                     const size_t* input_shape, const ptrdiff_t* input_strides,
                     const size_t* out_shape, const ptrdiff_t* out_strides,
                     size_t output_size, size_t hidden_size, int ndim,
                     bool input_contiguous, bool out_contiguous);

template <>
class Operator<SiluAndMul, Device::Type::kCambricon> : public SiluAndMul {
 public:
  Operator(const Tensor input, Tensor out) : SiluAndMul{input, out} {
    assert((out_type_ == DataType::kFloat16 ||
            out_type_ == DataType::kBFloat16 ||
            out_type_ == DataType::kFloat32) &&
           "`SiluAndMul` Cambricon supports float16, bfloat16, and float32");
    cnrt_utils::GetLaunchConfig(input.device(), &core_per_cluster_,
                                &cluster_count_);
    CNRT_CHECK(cnrtMalloc(&default_workspace_, workspace_size_in_bytes()));
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }

    auto queue = static_cast<cnrtQueue_t>(stream_ ? stream_ : 0);
    auto* workspace = workspace_ ? workspace_ : default_workspace_;
    DispatchFunc<
        Device::Type::kCambricon,
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          SiluAndMulUnion<T>(
              workspace, core_per_cluster_, cluster_count_, queue, input.data(),
              out.data(), input_shape_.data(), input_strides_.data(),
              out_shape_.data(), out_strides_.data(), output_size_,
              hidden_size_, static_cast<int>(ndim_), is_input_contiguous_,
              is_out_contiguous_);
        },
        "CambriconSiluAndMul::operator()");
  }

  ~Operator() { CNRT_CHECK(cnrtFree(default_workspace_)); }

  std::size_t workspace_size_in_bytes() const override {
    return ndim_ * (2 * sizeof(size_t) + 2 * sizeof(ptrdiff_t));
  }

 private:
  void* default_workspace_{nullptr};
  int core_per_cluster_{0};
  int cluster_count_{0};
};

}  // namespace infini::ops

#endif  // INFINI_OPS_CAMBRICON_SILU_AND_MUL_KERNEL_H_
