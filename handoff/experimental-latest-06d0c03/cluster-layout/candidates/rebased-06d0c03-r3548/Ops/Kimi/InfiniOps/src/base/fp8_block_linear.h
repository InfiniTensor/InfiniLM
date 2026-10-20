#pragma once
#include "base/moe_compute_common.h"

namespace infini::ops {
// E4M3FN payloads use U8 storage; scales retain their original global blocks.
// Activation quantization must occur before any TP slice crossing a block.
class Fp8BlockLinear : public Operator<Fp8BlockLinear> {
 public:
  Fp8BlockLinear(Tensor output,const Tensor x,const Tensor x_scale,const Tensor weight,
                 const Tensor weight_scale,size_t row_offset,size_t column_offset)
      : metadata_{output,x,x_scale,weight,weight_scale},row_offset_(row_offset),column_offset_(column_offset) {
    Validate(output,x,x_scale,weight,weight_scale,row_offset,column_offset);
  }
  virtual void operator()(Tensor output,const Tensor x,const Tensor x_scale,const Tensor weight,
                          const Tensor weight_scale,size_t row_offset,size_t column_offset) const = 0;
 protected:
  static void Validate(Tensor output,const Tensor x,const Tensor x_scale,const Tensor weight,
                       const Tensor weight_scale,size_t row_offset,size_t column_offset) {
    using namespace moe_detail;
    Require(output.ndim()==2 && x.ndim()==2 && weight.ndim()==2 && x_scale.ndim()==2 && weight_scale.ndim()==2);
    Require(x.dtype()==DataType::kUInt8 && weight.dtype()==DataType::kUInt8);
    Require(x_scale.dtype()==DataType::kFloat32 && weight_scale.dtype()==DataType::kFloat32);
    Require(output.dtype()==DataType::kFloat32 || Reduced(output.dtype()));
    const auto m=x.size(0), k=x.size(1), n=weight.size(0);
    // Native launch maps tokens to CUDA-compatible grid.y on gfx936.
    Require(m<=65535 && n<=size_t(std::numeric_limits<int32_t>::max()));
    Require(k && n && weight.size(1)==k && output.shape()==Tensor::Shape{m,n} && x_scale.size(0)==m);
    const auto rows=Product(weight_scale.size(0),128), columns=Product(weight_scale.size(1),128);
    Require(row_offset<=rows && n<=rows-row_offset && column_offset<=columns && k<=columns-column_offset);
    Require(x_scale.size(1)==weight_scale.size(1));
    Views({output,x,x_scale,weight,weight_scale},1);
  }
  void Invocation(Tensor output,const Tensor x,const Tensor x_scale,const Tensor weight,
                  const Tensor weight_scale,size_t row_offset,size_t column_offset) const {
    moe_detail::Same(metadata_,{output,x,x_scale,weight,weight_scale});
    moe_detail::Require(row_offset==row_offset_ && column_offset==column_offset_);
    Validate(output,x,x_scale,weight,weight_scale,row_offset,column_offset);
  }
  std::vector<Tensor> metadata_;
  size_t row_offset_,column_offset_;
};
} // namespace infini::ops
