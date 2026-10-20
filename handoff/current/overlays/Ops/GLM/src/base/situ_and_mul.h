#pragma once
#include "base/moe_compute_common.h"
#include <array>

namespace infini::ops {
class SituAndMul : public Operator<SituAndMul> {
 public:
  SituAndMul(Tensor output,const Tensor gate,const Tensor up,float beta,float linear_beta)
      : metadata_{output,gate,up},beta_(beta),linear_beta_(linear_beta) {
    Validate(output,gate,up,beta,linear_beta);
  }
  virtual void operator()(Tensor output,const Tensor gate,const Tensor up,float beta,float linear_beta) const=0;
 protected:
  static size_t Span(Tensor t) {
    using namespace moe_detail;
    Require(t.shape().size()==t.strides().size());
    size_t offset=0;
    for(size_t d=0;d<t.ndim();++d) {
      Require(t.stride(d)>=0);
      if(!t.size(d))return 0;
      auto term=Product(t.size(d)-1,size_t(t.stride(d)));
      Require(term<=std::numeric_limits<size_t>::max()-offset);offset+=term;
    }
    Require(offset<std::numeric_limits<size_t>::max());
    return Product(offset+1,t.element_size());
  }
  static void Validate(Tensor output,const Tensor gate,const Tensor up,float beta,float linear_beta) {
    using namespace moe_detail;
    Require(std::isfinite(beta)&&beta>0 && std::isfinite(linear_beta)&&linear_beta>0);
    Require(output.ndim()>0 && output.ndim()<=8 && output.IsContiguous());
    Require(output.shape()==gate.shape() && output.shape()==up.shape());
    Require(output.device()==gate.device() && output.device()==up.device());
    Require(output.dtype()==gate.dtype() && output.dtype()==up.dtype());
    Require(Reduced(output.dtype()) || output.dtype()==DataType::kFloat32);
    const auto bytes=Span(output),gbytes=Span(gate),ubytes=Span(up);
    for(const auto t:{output,gate,up})
      Require(!t.numel() || (t.data() && reinterpret_cast<uintptr_t>(t.data())%t.element_size()==0));
    auto clear=[&](Tensor t,size_t span) {
      auto a=reinterpret_cast<uintptr_t>(output.data()),b=reinterpret_cast<uintptr_t>(t.data());
      return !bytes || !span || (a<=b?b-a>=bytes:a-b>=span);
    };
    Require(clear(gate,gbytes) && clear(up,ubytes));
  }
  void Invocation(Tensor output,const Tensor gate,const Tensor up,float beta,float linear_beta) const {
    moe_detail::Same(metadata_,{output,gate,up});
    moe_detail::Require(beta==beta_ && linear_beta==linear_beta_);
    Validate(output,gate,up,beta,linear_beta);
  }
  const std::vector<Tensor> metadata_;
  const float beta_,linear_beta_;
};
} // namespace infini::ops
