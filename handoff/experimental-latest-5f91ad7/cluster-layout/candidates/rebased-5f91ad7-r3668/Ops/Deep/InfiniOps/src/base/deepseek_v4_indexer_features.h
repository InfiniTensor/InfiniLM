#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
// D128 indexer features: optional weighted RMSNorm, interleaved trailing64
// RoPE, normalized F32 Hadamard, E4M3FN and one F32 scale per vector.
// rope[T,64] stores cos0,sin0,cos1,sin1,... . No intermediate BF16 roundtrip.
// Status2/nonfinite failure leaves both feature outputs unchanged.
class DeepseekV4IndexerFeatures:public Operator<DeepseekV4IndexerFeatures>{
public:
 DeepseekV4IndexerFeatures(Tensor output,Tensor scale,const Tensor input,const Tensor norm,const Tensor rope,Tensor status,bool normalize,float eps):metadata_{output,scale,input,norm,rope,status},normalize_(normalize),eps_(eps){Validate(output,scale,input,norm,rope,status,normalize,eps);workspace_bytes_=input.numel()+input.size(0)*input.size(1)*4;}
 virtual void operator()(Tensor output,Tensor scale,const Tensor input,const Tensor norm,const Tensor rope,Tensor status,bool normalize,float eps)const=0;
 size_t workspace_size_in_bytes()const override{return workspace_bytes_;}
protected:
 static void Validate(Tensor output,Tensor scale,const Tensor input,const Tensor norm,const Tensor rope,Tensor status,bool,float eps){using namespace moe_detail;Require(input.ndim()==3&&input.size(0)&&input.size(0)<=4096&&input.size(1)&&input.size(1)<=128&&input.size(2)==128&&input.dtype()==DataType::kBFloat16);Require(output.shape()==input.shape()&&output.dtype()==DataType::kUInt8&&scale.shape()==Tensor::Shape{input.size(0),input.size(1)}&&scale.dtype()==DataType::kFloat32);Require(norm.shape()==Tensor::Shape{128}&&norm.dtype()==DataType::kBFloat16&&rope.shape()==Tensor::Shape{input.size(0),64}&&rope.dtype()==DataType::kFloat32);Require(status.shape()==Tensor::Shape{1}&&status.dtype()==DataType::kInt32&&std::isfinite(eps)&&eps>0);Views({output,scale,status,input,norm,rope},3);}
 void Invocation(Tensor output,Tensor scale,const Tensor input,const Tensor norm,const Tensor rope,Tensor status,bool normalize,float eps)const{moe_detail::Same(metadata_,{output,scale,input,norm,rope,status});moe_detail::Require(normalize==normalize_&&eps==eps_);Validate(output,scale,input,norm,rope,status,normalize,eps);}
 std::vector<Tensor>metadata_;bool normalize_;float eps_;size_t workspace_bytes_;
};
}
