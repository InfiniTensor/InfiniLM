#pragma once
#include "base/indexed_kimi_delta_attention.h"
#include <cuda_runtime.h>
namespace infini::ops::indexed_kda_detail {
struct Layout {size_t requests,tokens,heads,dim,pool;ptrdiff_t strides[8][4]{};bool i64[3],norm;float scale,lower;};
void Launch(DataType,DataType,void*,void*,void*,const void*,const void*,const void*,const void*,const void*,
    const float*,const float*,const void*,const void*,const void*,Layout,cudaStream_t);
}
namespace infini::ops {
template<> class Operator<IndexedKimiDeltaAttention,Device::Type::kHygon> : public IndexedKimiDeltaAttention {
 public:
  using IndexedKimiDeltaAttention::IndexedKimiDeltaAttention;
  bool capture_safe() const override{return false;}
  static bool supports(Tensor out,Tensor state,const Tensor q,const Tensor k,const Tensor v,
      const Tensor gate,const Tensor beta,const Tensor alog,const Tensor bias,
      const Tensor offsets,const Tensor reads,const Tensor writes,float scale,float lower,bool norm) {
    try {Validate(out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes,scale,lower,norm);return out.device().type()==Device::Type::kHygon;}
    catch(const std::invalid_argument&){return false;}
  }
  void operator()(Tensor out,Tensor state,const Tensor q,const Tensor k,const Tensor v,
      const Tensor gate,const Tensor beta,const Tensor alog,const Tensor bias,
      const Tensor offsets,const Tensor reads,const Tensor writes,float scale,float lower,bool norm) const override {
    Invocation(out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes,scale,lower,norm);
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess||device!=out.device().index())throw std::invalid_argument("KDA owning device mismatch");
    if(!workspace_||workspace_size_in_bytes_<workspace_bytes_||reinterpret_cast<uintptr_t>(workspace_)%256)throw std::invalid_argument("KDA caller workspace missing or unaligned");
    // Cached operators retain geometry, but operand addresses change on every
    // invocation. Check the live tensors rather than stale constructor views.
    for(auto tensor:{out,state,q,k,v,gate,beta,alog,bias,offsets,reads,writes}){auto span=Span(tensor);auto ap=reinterpret_cast<uintptr_t>(workspace_),bp=reinterpret_cast<uintptr_t>(tensor.data());
      if(span&&(ap<=bp?bp-ap<workspace_bytes_:ap-bp<span))throw std::invalid_argument("KDA workspace aliases operands");}
    indexed_kda_detail::Layout l{};l.requests=reads.numel();l.tokens=q.size(1);l.heads=q.size(2);l.dim=q.size(3);l.pool=state.size(0);
    size_t i=0;for(auto tensor:{out,state,q,k,v,gate,beta,bias}){for(size_t j=0;j<tensor.ndim();++j)l.strides[i][j]=tensor.stride(j);++i;}
    l.i64[0]=offsets.dtype()==DataType::kInt64;l.i64[1]=reads.dtype()==DataType::kInt64;l.i64[2]=writes.dtype()==DataType::kInt64;
    l.norm=norm;l.scale=scale;l.lower=lower;
    indexed_kda_detail::Launch(q.dtype(),gate.dtype(),out.data(),state.data(),workspace_,q.data(),k.data(),v.data(),gate.data(),beta.data(),
      reinterpret_cast<const float*>(alog.data()),reinterpret_cast<const float*>(bias.data()),offsets.data(),reads.data(),writes.data(),l,reinterpret_cast<cudaStream_t>(stream_));
  }
};
}
