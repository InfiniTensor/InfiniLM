#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
// Query weights already contain q_scale/sqrt(128*global_heads).
// scores[t,k]=sum_h weights[t,h]*relu(dot(q_fp8[t,h],k_fp8[k])*k_scale[k]).
// Causal visibility is an explicit per-query compressed-prefix length.
class DeepseekV4IndexerScores:public Operator<DeepseekV4IndexerScores>{
public:
 DeepseekV4IndexerScores(Tensor output,const Tensor query,const Tensor weights,const Tensor key,const Tensor scales,const Tensor visible,Tensor status):metadata_{output,query,weights,key,scales,visible,status}{Validate(output,query,weights,key,scales,visible,status);}
 virtual void operator()(Tensor output,const Tensor query,const Tensor weights,const Tensor key,const Tensor scales,const Tensor visible,Tensor status)const=0;
 size_t workspace_size_in_bytes()const override{return 4;}
protected:
 static void Validate(Tensor output,const Tensor query,const Tensor weights,const Tensor key,const Tensor scales,const Tensor visible,Tensor status){using namespace moe_detail;Require(query.ndim()==3&&query.size(0)&&query.size(0)<=4096&&query.size(1)&&query.size(1)<=128&&query.size(2)==128&&query.dtype()==DataType::kUInt8);Require(key.ndim()==2&&key.size(0)&&key.size(0)<=262144&&key.size(1)==128&&key.dtype()==DataType::kUInt8);Require(weights.shape()==Tensor::Shape{query.size(0),query.size(1)}&&weights.dtype()==DataType::kFloat32&&scales.shape()==Tensor::Shape{key.size(0),1}&&scales.dtype()==DataType::kFloat32);Require(output.shape()==Tensor::Shape{query.size(0),key.size(0)}&&output.dtype()==DataType::kFloat32&&visible.shape()==Tensor::Shape{query.size(0)}&&visible.dtype()==DataType::kInt32&&status.shape()==Tensor::Shape{1}&&status.dtype()==DataType::kInt32);Views({output,status,query,weights,key,scales,visible},2);}
 void Invocation(Tensor output,const Tensor query,const Tensor weights,const Tensor key,const Tensor scales,const Tensor visible,Tensor status)const{moe_detail::Same(metadata_,{output,query,weights,key,scales,visible,status});Validate(output,query,weights,key,scales,visible,status);}
 std::vector<Tensor>metadata_;
};
}
