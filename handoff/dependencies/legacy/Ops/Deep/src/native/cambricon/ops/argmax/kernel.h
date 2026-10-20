#ifndef INFINI_OPS_CAMBRICON_ARGMAX_KERNEL_H_
#define INFINI_OPS_CAMBRICON_ARGMAX_KERNEL_H_
#include <climits>
#include <cstdint>
#include <stdexcept>
#include "base/argmax.h"
#include "native/cambricon/common.h"
namespace infini::ops {
void CambriconArgmaxLaunch(cnrtQueue_t,const void*,void*,int32_t,std::size_t,uint32_t);
template<> class Operator<Argmax,Device::Type::kCambricon>:public Argmax {
 public:
  static bool supports(const Tensor x,const std::optional<int64_t> dim,
                       bool keepdim,const Tensor y) {
    if (x.device().type()!=Device::Type::kCambricon || x.device().index()<0 ||
        x.dtype()!=DataType::kFloat32 || x.ndim()!=1 || !x.numel() ||
        x.numel()>INT_MAX || !x.IsContiguous() || !x.data() || dim || keepdim ||
        x.device()!=y.device() || y.ndim()!=0 || y.dtype()!=DataType::kInt64 ||
        !y.IsContiguous() || !y.data()) return false;
    const auto a=reinterpret_cast<uintptr_t>(x.data()),b=reinterpret_cast<uintptr_t>(y.data());
    if (a%sizeof(float) || b%alignof(int64_t)) return false;
    return b>a?b-a>=x.numel()*sizeof(float):a-b>=sizeof(int64_t);
  }
  Operator(const Tensor x,const std::optional<int64_t> dim,bool keepdim,Tensor y)
      :Argmax(x,dim,keepdim,y) {}
  void operator()(const Tensor x,const std::optional<int64_t> dim,
                  bool keepdim,Tensor y)const override {
    if (!supports(x,dim,keepdim,y) || x.shape()!=input_shape_ ||
        x.strides()!=input_strides_ || x.dtype()!=input_type_ ||
        y.shape()!=out_shape_ || y.strides()!=out_strides_ || y.dtype()!=out_type_ ||
        x.device().index()!=device_index_ || dim!=dim_ || keepdim!=keepdim_)
      throw std::invalid_argument("Cambricon Argmax prepared FP32 metadata changed");
    CambriconArgmaxLaunch(static_cast<cnrtQueue_t>(stream_),x.data(),y.data(),
                         static_cast<int32_t>(x.numel()),sizeof(float),0x7f800000U);
    CNRT_CHECK(cnrtGetLastError());
  }
};
} // namespace infini::ops
#endif
