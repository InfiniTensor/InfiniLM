#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
class BlockQuantF8E4m3 : public Operator<BlockQuantF8E4m3> {
 public:
  BlockQuantF8E4m3(const Tensor input,Tensor packed,Tensor scale):metadata_{input,packed,scale}{Validate(input,packed,scale);}
  virtual void operator()(const Tensor input,Tensor packed,Tensor scale) const=0;
 protected:
  static void Validate(const Tensor input,Tensor packed,Tensor scale){
    using namespace moe_detail;
    Require(input.ndim()==2&&(Reduced(input.dtype())||input.dtype()==DataType::kFloat32));
    Require(input.size(0)&&input.size(0)<=65535&&input.size(1)&&input.size(1)<=65535*128ULL);
    Require(packed.shape()==input.shape()&&packed.dtype()==DataType::kUInt8);
    Require(scale.shape()==Tensor::Shape{input.size(0),(input.size(1)+127)/128}&&scale.dtype()==DataType::kFloat32);
    Views({packed,scale,input},2);
  }
  void Invocation(const Tensor input,Tensor packed,Tensor scale) const {
    moe_detail::Same(metadata_,{input,packed,scale});Validate(input,packed,scale);
  }
  std::vector<Tensor> metadata_;
};
}
