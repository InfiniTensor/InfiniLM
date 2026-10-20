#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
// mode0 packs BF16 D512 into V4.0 pages; mode1 gathers decoded BF16 values.
// status bits: 1=slot bounds, 2=duplicate write, 4=nonfinite source. A failed
// validation leaves the cache (store) or values (gather) completely unchanged.
class DeepseekV4PagedCache:public Operator<DeepseekV4PagedCache>{
public:
 DeepseekV4PagedCache(Tensor cache,const Tensor slots,Tensor values,Tensor status,size_t block,size_t mode):metadata_{cache,slots,values,status},block_(block),mode_(mode){Validate(cache,slots,values,status,block,mode);workspace_bytes_=mode==0?((cache.size(0)*block+31)/32)*4:0;}
 virtual void operator()(Tensor cache,const Tensor slots,Tensor values,Tensor status,size_t block,size_t mode)const=0;
 size_t workspace_size_in_bytes()const override{return workspace_bytes_;}
protected:
 static void Validate(Tensor cache,const Tensor slots,Tensor values,Tensor status,size_t block,size_t mode){using namespace moe_detail;
  Require(mode<=1&&(block==2||block==64||block==256));Require(cache.ndim()==2&&cache.dtype()==DataType::kUInt8&&cache.size(0)&&cache.size(0)<=65536&&cache.size(1)>=block*584&&cache.size(1)%2==0);
  Require(slots.ndim()==1&&slots.dtype()==DataType::kInt64&&slots.size(0)&&slots.size(0)<=65536);Require(values.shape()==Tensor::Shape{slots.size(0),512}&&values.dtype()==DataType::kBFloat16);Require(status.shape()==Tensor::Shape{1}&&status.dtype()==DataType::kInt32);Views({cache,slots,values,status},0);
 }
 void Invocation(Tensor cache,const Tensor slots,Tensor values,Tensor status,size_t block,size_t mode)const{moe_detail::Same(metadata_,{cache,slots,values,status});moe_detail::Require(block==block_&&mode==mode_);Validate(cache,slots,values,status,block,mode);}
 std::vector<Tensor>metadata_;size_t block_,mode_,workspace_bytes_;
};
}
