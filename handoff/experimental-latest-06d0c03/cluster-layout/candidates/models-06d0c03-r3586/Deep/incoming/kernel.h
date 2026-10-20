#ifndef INFINI_OPS_HYGON_TOPK_SIGMOID_KERNEL_H_
#define INFINI_OPS_HYGON_TOPK_SIGMOID_KERNEL_H_

#include <optional>

#include "base/topk_sigmoid.h"
#include "native/cuda/hygon/ops/binary_view_capture.h"

namespace infini::ops {

template <>
class Operator<TopkSigmoid, Device::Type::kHygon, 0> : public TopkSigmoid {
 public:
  static bool supports(const Tensor g,std::optional<Tensor> b,std::optional<Tensor> p,
      bool renormalize,double factor,Tensor w,Tensor ids,Tensor scratch) {
    using namespace hygon_binary_view;
    if(g.device().type()!=Device::Type::kHygon || g.ndim()!=2 || w.ndim()!=2 ||
       !g.size(1) || !w.size(1) || w.size(1)>g.size(1) || g.size(0)>65535 ||
       g.size(1)>INT32_MAX || w.size(1)>INT32_MAX ||
       g.size(0)>size_t(INT32_MAX)/w.size(1) ||
       w.shape()!=Tensor::Shape{g.size(0),w.size(1)} || ids.shape()!=w.shape() || scratch.shape()!=w.shape() ||
       w.dtype()!=DataType::kFloat32 || ids.dtype()!=DataType::kInt32 || scratch.dtype()!=DataType::kInt32 ||
       (g.dtype()!=DataType::kFloat32 && g.dtype()!=DataType::kFloat16 && g.dtype()!=DataType::kBFloat16) ||
       !std::isfinite(factor) || factor<0 || factor>std::numeric_limits<float>::max())return false;
    std::vector<Tensor> tensors{g,w,ids,scratch};
    if(b) {if(b->shape()!=Tensor::Shape{g.size(1)} || b->dtype()!=DataType::kFloat32)return false;tensors.push_back(*b);}
    if(p) {if(p->shape()!=Tensor::Shape{g.size(0)} || p->dtype()!=DataType::kBool)return false;tensors.push_back(*p);}
    for(size_t i=0;i<tensors.size();++i) {
      auto t=tensors[i];size_t span=0;
      if(t.device()!=g.device() || !t.IsContiguous() || !SpanBytes(t,&span) ||
         (span && reinterpret_cast<uintptr_t>(t.data())%t.element_size()))return false;
      for(size_t j=0;j<i;++j) {
        if((i>=1 && i<=3)||(j>=1 && j<=3)) {
          size_t other=0;if(!SpanBytes(tensors[j],&other)||Overlaps(t,span,tensors[j],other))return false;
        }
      }
    }
    return true;
  }
  static Tensor Checked(Tensor g,std::optional<Tensor> b,std::optional<Tensor> p,
      bool r,double f,Tensor w,Tensor i,Tensor s) {
    if(!supports(g,b,p,r,f,w,i,s))throw std::invalid_argument("unsupported Hygon sigmoid routing operands");
    return g;
  }
  Operator(Tensor g,std::optional<Tensor> b,std::optional<Tensor> p,
      bool r,double f,Tensor w,Tensor i,Tensor s)
      :TopkSigmoid(Checked(g,b,p,r,f,w,i,s),b,p,r,f,w,i,s) {}
  bool capture_safe() const override { return false; }

  void operator()(const Tensor gating_output,
                  std::optional<Tensor> e_score_correction_bias,
                  std::optional<Tensor> is_padding, const bool renormalize,
                  const double routed_scaling_factor, Tensor topk_weights,
                  Tensor topk_ids, Tensor token_expert_indices) const override;
};

}  // namespace infini::ops

#endif  // INFINI_OPS_HYGON_TOPK_SIGMOID_KERNEL_H_
