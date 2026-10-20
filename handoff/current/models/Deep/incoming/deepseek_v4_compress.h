#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
// Stateful temporal softmax pooling. C4 consumes previous/current four-token
// windows; C128 consumes one 128-token window. Output is zero off boundaries.
// The caller supplies valid previous state. All outputs read that snapshot
// before final writes are committed. Status1 is metadata, status2 nonfinite;
// validation failure preserves both output and state, including padding.
class DeepseekV4Compress:public Operator<DeepseekV4Compress>{
public:
 DeepseekV4Compress(Tensor output,const Tensor score,const Tensor ape,Tensor state,const Tensor positions,const Tensor writes,const Tensor extras,Tensor status,size_t ratio):metadata_{output,score,ape,state,positions,writes,extras,status},ratio_(ratio){Validate(output,score,ape,state,positions,writes,extras,status,ratio);workspace_bytes_=state.size(0)*sizeof(int32_t);}
 virtual void operator()(Tensor output,const Tensor score,const Tensor ape,Tensor state,const Tensor positions,const Tensor writes,const Tensor extras,Tensor status,size_t ratio)const=0;
 size_t workspace_size_in_bytes()const override{return workspace_bytes_;}
protected:
 static void Validate(Tensor output,const Tensor score,const Tensor ape,Tensor state,const Tensor positions,const Tensor writes,const Tensor extras,Tensor status,size_t ratio){using namespace moe_detail;
  Require(ratio==4||ratio==128);Require(output.ndim()==2&&output.dtype()==DataType::kBFloat16&&output.size(0)&&output.size(0)<=4096&&(output.size(1)==128||output.size(1)==512));const auto t=output.size(0),d=output.size(1),channels=ratio==4?size_t{4}:size_t{2};
  Require(score.shape()==Tensor::Shape{t,size_t(channels)*d}&&score.dtype()==DataType::kBFloat16);Require(ape.shape()==Tensor::Shape{ratio==4?size_t(8):size_t(128),d}&&ape.dtype()==DataType::kFloat32);
  Require(state.ndim()==2&&state.size(0)&&state.size(0)<=65536&&state.size(0)%ratio==0&&state.size(1)==size_t(channels)*d&&(state.dtype()==DataType::kFloat32||state.dtype()==DataType::kBFloat16));
  for(auto tensor:{positions,writes,extras})Require(tensor.shape()==Tensor::Shape{t}&&tensor.dtype()==DataType::kInt64);Require(status.shape()==Tensor::Shape{1}&&status.dtype()==DataType::kInt32);Views({output,score,ape,state,positions,writes,extras,status},0);
 }
 void Invocation(Tensor output,const Tensor score,const Tensor ape,Tensor state,const Tensor positions,const Tensor writes,const Tensor extras,Tensor status,size_t ratio)const{moe_detail::Same(metadata_,{output,score,ape,state,positions,writes,extras,status});moe_detail::Require(ratio==ratio_);Validate(output,score,ape,state,positions,writes,extras,status,ratio);}
 std::vector<Tensor>metadata_;size_t ratio_,workspace_bytes_;
};
}
