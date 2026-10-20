#pragma once
#include "base/moe_compute_common.h"
#include <set>
namespace infini::ops {
// Packed varlen q/k/v [1,T,H,D]; state pool [P,H,D,D] stores [value,key].
class IndexedKimiDeltaAttention : public Operator<IndexedKimiDeltaAttention> {
 public:
  IndexedKimiDeltaAttention(Tensor out,Tensor state,const Tensor q,const Tensor k,const Tensor v,
      const Tensor gate,const Tensor beta,const Tensor alog,const Tensor bias,
      const Tensor offsets,const Tensor reads,const Tensor writes,float scale,float lower,bool norm)
      :metadata_{out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes},scale_(scale),lower_(lower),norm_(norm) {
    Validate(out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes,scale,lower,norm);
    workspace_bytes_=moe_detail::Product(state.numel(),state.element_size());
  }
  virtual void operator()(Tensor out,Tensor state,const Tensor q,const Tensor k,const Tensor v,
      const Tensor gate,const Tensor beta,const Tensor alog,const Tensor bias,
      const Tensor offsets,const Tensor reads,const Tensor writes,float scale,float lower,bool norm) const=0;
  size_t workspace_size_in_bytes() const override{return workspace_bytes_;}
 protected:
  static size_t Span(Tensor t) {
    using namespace moe_detail;Require(t.ndim()==t.strides().size());size_t span=0;
    for(size_t i=0;i<t.ndim();++i){Require(t.stride(i)>=0);if(!t.size(i))return 0;
      auto term=Product(t.size(i)-1,size_t(t.stride(i)));Require(term<=SIZE_MAX-span);span+=term;}
    Require(span<SIZE_MAX);return Product(span+1,t.element_size());
  }
  static bool Overlap(Tensor a,Tensor b) {
    auto an=Span(a),bn=Span(b);auto ap=reinterpret_cast<uintptr_t>(a.data()),bp=reinterpret_cast<uintptr_t>(b.data());
    return an&&bn&&(ap<=bp?bp-ap<an:ap-bp<bn);
  }
  static void Validate(Tensor out,Tensor state,const Tensor q,const Tensor k,const Tensor v,
      const Tensor gate,const Tensor beta,const Tensor alog,const Tensor bias,
      const Tensor offsets,const Tensor reads,const Tensor writes,float scale,float lower,bool norm) {
    using namespace moe_detail;
    Require(q.ndim()==4&&q.size(0)==1&&state.ndim()==4&&beta.ndim()==3);
    const auto t=q.size(1),h=q.size(2),d=q.size(3),p=state.size(0);
    Require(p&&h&&h<=65535&&d&&d<=128&&t<=size_t(INT32_MAX));
    Product(Product(Product(p,h),d),d);
    Require(out.shape()==q.shape()&&k.shape()==q.shape()&&v.shape()==q.shape()&&gate.shape()==q.shape());
    Require(beta.shape()==Tensor::Shape{1,t,h}&&state.shape()==Tensor::Shape{p,h,d,d});
    Require(out.dtype()==q.dtype()&&state.dtype()==q.dtype()&&k.dtype()==q.dtype()&&v.dtype()==q.dtype());
    Require(Reduced(q.dtype())||q.dtype()==DataType::kFloat32);
    Require(beta.dtype()==gate.dtype()&&(Reduced(gate.dtype())||gate.dtype()==DataType::kFloat32));
    Require(alog.shape()==Tensor::Shape{h}&&bias.shape()==Tensor::Shape{h,d}&&alog.dtype()==DataType::kFloat32&&bias.dtype()==DataType::kFloat32);
    Require(reads.ndim()==1&&writes.shape()==reads.shape()&&reads.numel()&&reads.numel()<=65535);
    Require(offsets.shape()==Tensor::Shape{reads.numel()+1});
    for(auto tensor:{offsets,reads,writes})Require(tensor.dtype()==DataType::kInt32||tensor.dtype()==DataType::kInt64);
    Require(out.IsContiguous()&&state.IsContiguous()&&alog.IsContiguous()&&bias.IsContiguous()&&offsets.IsContiguous()&&reads.IsContiguous()&&writes.IsContiguous());
    for(auto tensor:{q,k,v,gate})Require(tensor.stride(3)==1);
    Require(std::isfinite(scale)&&scale>0&&std::isfinite(lower)&&lower<=0);
    std::vector<Tensor> tensors{out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes};
    for(size_t i=0;i<tensors.size();++i){auto tensor=tensors[i];auto span=Span(tensor);
      Require(tensor.device()==out.device()&&(!span||(tensor.data()&&reinterpret_cast<uintptr_t>(tensor.data())%tensor.element_size()==0)));
      for(size_t j=0;j<i&&j<2;++j)Require(!Overlap(tensor,tensors[j]));}
  }
  void Invocation(Tensor out,Tensor state,const Tensor q,const Tensor k,const Tensor v,
      const Tensor gate,const Tensor beta,const Tensor alog,const Tensor bias,
      const Tensor offsets,const Tensor reads,const Tensor writes,float scale,float lower,bool norm) const {
    moe_detail::Same(metadata_,{out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes});
    moe_detail::Require(scale==scale_&&lower==lower_&&norm==norm_);
    Validate(out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes,scale,lower,norm);
  }
  const std::vector<Tensor> metadata_;float scale_,lower_;bool norm_;size_t workspace_bytes_{};
};
}
