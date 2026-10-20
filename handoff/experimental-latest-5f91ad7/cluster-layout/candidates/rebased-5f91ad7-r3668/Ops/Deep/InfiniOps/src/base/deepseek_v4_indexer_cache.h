#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
// C4 indexer page: 64 rows of128 E4M3 bytes, then64 F32 vector scales.
// mode0 stores, mode1 gathers. Status1=bounds,2=duplicate,4=nonfinite.
class DeepseekV4IndexerCache:public Operator<DeepseekV4IndexerCache>{
public:
 DeepseekV4IndexerCache(Tensor cache,const Tensor slots,Tensor values,Tensor scales,Tensor status,size_t mode):metadata_{cache,slots,values,scales,status},mode_(mode){Validate(cache,slots,values,scales,status,mode);workspace_bytes_=cache.size(0)*8;}
 virtual void operator()(Tensor cache,const Tensor slots,Tensor values,Tensor scales,Tensor status,size_t mode)const=0;
 size_t workspace_size_in_bytes()const override{return workspace_bytes_;}
protected:
 static void Validate(Tensor cache,const Tensor slots,Tensor values,Tensor scales,Tensor status,size_t mode){using namespace moe_detail;Require(mode<=1&&cache.ndim()==2&&cache.dtype()==DataType::kUInt8&&cache.size(0)&&cache.size(0)<=65536&&cache.size(1)>=8448&&cache.size(1)%4==0);Require(slots.ndim()==1&&slots.size(0)&&slots.size(0)<=65536&&slots.dtype()==DataType::kInt64);Require(values.shape()==Tensor::Shape{slots.size(0),128}&&values.dtype()==DataType::kUInt8&&scales.shape()==Tensor::Shape{slots.size(0),1}&&scales.dtype()==DataType::kFloat32&&status.shape()==Tensor::Shape{1}&&status.dtype()==DataType::kInt32);Views({cache,slots,values,scales,status},5);}
 void Invocation(Tensor cache,const Tensor slots,Tensor values,Tensor scales,Tensor status,size_t mode)const{moe_detail::Same(metadata_,{cache,slots,values,scales,status});moe_detail::Require(mode==mode_);Validate(cache,slots,values,scales,status,mode);}
 std::vector<Tensor>metadata_;size_t mode_,workspace_bytes_;
};
}
