#pragma once
#include <cmath>
#include <cuda_runtime_api.h>
#include "native/cuda/hygon/caster.cuh"
#include "native/cuda/hygon/runtime_.h"
#include "native/cuda/hygon/ops/binary_view_capture.h"
#include "native/cuda/ops/fill/kernel.h"
namespace infini::ops {
template<> class Operator<Fill,Device::Type::kHygon,0>
    :public CudaFill<Runtime<Device::Type::kHygon>> {
 public:
  using Parent=CudaFill<Runtime<Device::Type::kHygon>>;
  using Parent::Parent;
  static bool supports(const Tensor input,const double value,Tensor out){
    if(input.device().type()!=Device::Type::kHygon || !Fill::supports(input,value,out)
       || !hygon_binary_view::NonOverlappingOutput(out))return false;
    int device=-1;
    if(cudaGetDevice(&device)!=cudaSuccess || device!=out.device().index())return false;
    for(auto stride:out.strides())if(stride<0)return false;
    return out.dtype()==DataType::kFloat16 || out.dtype()==DataType::kBFloat16
        || out.dtype()==DataType::kFloat32 || out.dtype()==DataType::kFloat64
        || ((out.dtype()==DataType::kInt32 || out.dtype()==DataType::kInt64)
            && std::isfinite(value) && value>=-2147483648. && value<=2147483647.);
  }
  bool capture_safe()const override{return false;}
};
}
