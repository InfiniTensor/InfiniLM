#pragma once
#include "base/moe_compute_common.h"

namespace infini::ops {
struct Mxfp4MoeGeometry {
  size_t num_tokens, topk, num_experts, hidden_size, intermediate_size, activation;
  size_t routeCount() const { return num_tokens * topk; }
};

class FusedMoeMxfp4 : public Operator<FusedMoeMxfp4> {
 public:
  FusedMoeMxfp4(Tensor output, const Tensor input, const Tensor indices,
                const Tensor routing_weights, const Tensor w13, const Tensor s13,
                const Tensor w2, const Tensor s2, size_t activation)
      : metadata_{output,input,indices,routing_weights,w13,s13,w2,s2},
        dtype_(input.dtype()) {
    Validate(output,input,indices,routing_weights,w13,s13,w2,s2,activation);
    geometry_={input.size(0),indices.size(1),w13.size(0),input.size(1),w2.size(2)*2,activation};
    workspace_bytes_=moe_detail::Product(moe_detail::Product(geometry_.routeCount(),geometry_.intermediate_size),input.element_size());
  }
  virtual void operator()(Tensor output, const Tensor input, const Tensor indices,
                         const Tensor routing_weights, const Tensor w13, const Tensor s13,
                         const Tensor w2, const Tensor s2, size_t activation) const = 0;
  size_t workspace_size_in_bytes() const override { return workspace_bytes_; }
 protected:
  static void Validate(Tensor output, const Tensor input, const Tensor indices,
                       const Tensor routing_weights, const Tensor w13, const Tensor s13,
                       const Tensor w2, const Tensor s2, size_t activation) {
    using namespace moe_detail;
    Require(input.ndim()==2 && indices.ndim()==2 && routing_weights.ndim()==2);
    Require(w13.ndim()==3 && s13.ndim()==3 && w2.ndim()==3 && s2.ndim()==3);
    Require(Reduced(input.dtype()) || input.dtype()==DataType::kFloat32);
    Require(output.dtype()==input.dtype() && output.shape()==input.shape());
    Require(indices.dtype()==DataType::kInt32 && routing_weights.dtype()==DataType::kFloat32);
    Require(indices.size(0)==input.size(0) && routing_weights.shape()==indices.shape());
    Require(w13.dtype()==DataType::kUInt8 && s13.dtype()==DataType::kUInt8 && w2.dtype()==DataType::kUInt8 && s2.dtype()==DataType::kUInt8);
    const auto h=input.size(1), i=w2.size(2)*2, e=w13.size(0), k=indices.size(1);
    Require(h && i && e && k && k<=e && h%32==0 && i%32==0 && (activation==1 || activation==2));
    Require(w13.shape()==Tensor::Shape{e,2*i,h/2} && s13.shape()==Tensor::Shape{e,2*i,h/32});
    Require(w2.shape()==Tensor::Shape{e,h,i/2} && s2.shape()==Tensor::Shape{e,h,i/32});
    Product(Product(input.size(0),k),i); Product(input.size(0),h);
    Views({output,input,indices,routing_weights,w13,s13,w2,s2},1);
  }
  void Invocation(Tensor output, const Tensor input, const Tensor indices,
                  const Tensor routing_weights, const Tensor w13, const Tensor s13,
                  const Tensor w2, const Tensor s2, size_t activation) const {
    moe_detail::Same(metadata_,{output,input,indices,routing_weights,w13,s13,w2,s2});
    moe_detail::Require(activation==geometry_.activation);
    Validate(output,input,indices,routing_weights,w13,s13,w2,s2,activation);
  }
  std::vector<Tensor> metadata_;
  DataType dtype_;
  Mxfp4MoeGeometry geometry_{};
  size_t workspace_bytes_{0};
};
} // namespace infini::ops
