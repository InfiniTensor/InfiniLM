#pragma once
#include "base/block_quant_f8_e4m3.h"
#include <cuda_runtime.h>
namespace infini::ops::block_quant_f8_detail {
void Launch(DataType,const void*,uint8_t*,float*,size_t,size_t,cudaStream_t);
}
namespace infini::ops {
template<>class Operator<BlockQuantF8E4m3,Device::Type::kHygon> : public BlockQuantF8E4m3 {
 public:
  using BlockQuantF8E4m3::BlockQuantF8E4m3;
  bool capture_safe() const override{return false;}
  void operator()(const Tensor input,Tensor packed,Tensor scale) const override {
    Invocation(input,packed,scale);int device=-1;
    if(input.device().type()!=Device::Type::kHygon||cudaGetDevice(&device)!=cudaSuccess||device!=input.device().index())throw std::invalid_argument("FP8 block quant requires owning Hygon device");
    block_quant_f8_detail::Launch(input.dtype(),input.data(),static_cast<uint8_t*>(packed.data()),static_cast<float*>(scale.data()),input.size(0),input.size(1),static_cast<cudaStream_t>(stream_));
  }
};
}
