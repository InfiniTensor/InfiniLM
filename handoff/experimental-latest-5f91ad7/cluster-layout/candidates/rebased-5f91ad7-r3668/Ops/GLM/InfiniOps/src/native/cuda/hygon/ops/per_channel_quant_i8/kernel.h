#pragma once
#include "base/per_channel_quant_i8.h"
#include "native/cuda/hygon/ops/binary_view_capture.h"
#include "native/cuda/hygon/runtime_.h"
#include <stdexcept>
namespace infini::ops::hygon_per_channel_i8_detail {
void Launch(const void*,std::int8_t*,float*,DataType,std::size_t,std::size_t,cudaStream_t);
}
namespace infini::ops {
template <> class Operator<PerChannelQuantI8,Device::Type::kHygon,0>
    : public PerChannelQuantI8 {
 public:
  static bool supports(const Tensor x,Tensor packed,Tensor scale) {
    using namespace hygon_binary_view;
    if(x.device().type()!=Device::Type::kHygon || x.device()!=packed.device() ||
       x.device()!=scale.device() || x.ndim()!=2 || x.shape()!=packed.shape() ||
       !x.IsContiguous() || !packed.IsContiguous() || !scale.IsContiguous() ||
       scale.shape()!=Tensor::Shape{x.shape()[0],1} ||
       packed.dtype()!=DataType::kInt8 || scale.dtype()!=DataType::kFloat32 ||
       (x.dtype()!=DataType::kFloat16 && x.dtype()!=DataType::kBFloat16 &&
        x.dtype()!=DataType::kFloat32) || !x.shape()[1]) return false;
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess || device!=x.device().index()) return false;

    std::size_t xb=0,pb=0,sb=0;
    if(!SpanBytes(x,&xb)||!SpanBytes(packed,&pb)||!SpanBytes(scale,&sb)) return false;
    if((xb && reinterpret_cast<std::uintptr_t>(x.data())%x.element_size()) ||
       (sb && reinterpret_cast<std::uintptr_t>(scale.data())%sizeof(float))) return false;
    return !Overlaps(x,xb,packed,pb)&&!Overlaps(x,xb,scale,sb)&&!Overlaps(packed,pb,scale,sb);
  }
  Operator(const Tensor x,Tensor packed,Tensor scale)
      : PerChannelQuantI8(x,packed,scale) {}
  bool capture_safe() const override { return false; }
  void operator()(const Tensor x,Tensor packed,Tensor scale) const override {
    if(!supports(x,packed,scale))
      throw std::invalid_argument("unsupported per-channel INT8 operands");
    hygon_per_channel_i8_detail::Launch(x.data(),static_cast<std::int8_t*>(packed.data()),
        static_cast<float*>(scale.data()),x.dtype(),x.shape()[0],x.shape()[1],
        static_cast<cudaStream_t>(stream_?stream_:0));
  }
};
}
