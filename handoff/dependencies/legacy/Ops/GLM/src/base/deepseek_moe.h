#pragma once
#include "base/moe_compute_common.h"
namespace infini::ops {
class DeepseekMoe : public Operator<DeepseekMoe> {
 public:

  static std::vector<Tensor> Collect(Tensor o,Tensor h,Tensor i,Tensor w,const std::vector<Tensor>& g,const std::vector<Tensor>& u,const std::vector<Tensor>& d) {
    std::vector<Tensor> v{o,h,i,w}; for(const auto& t:g) v.push_back(t); for(const auto& t:u) v.push_back(t); for(const auto& t:d) v.push_back(t); return v;
  }

  DeepseekMoe(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) : metadata_{Collect(output,hidden,ids,weights,gate,up,down)}, intermediate_(intermediate), num_experts_(num_experts) { Validate(output, hidden, ids, weights, gate, up, down, intermediate, num_experts); }
  static bool supports(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) { try { Validate(output, hidden, ids, weights, gate, up, down, intermediate, num_experts); return true; } catch(const std::invalid_argument&) { return false; } }
  virtual void operator()(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) const = 0;
 protected:
  void Invocation(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) const {
    moe_detail::Require(intermediate==intermediate_ && num_experts==num_experts_); moe_detail::Same(metadata_,Collect(output,hidden,ids,weights,gate,up,down)); Validate(output, hidden, ids, weights, gate, up, down, intermediate, num_experts);
  }
  static void Validate(Tensor output, const Tensor hidden, const Tensor ids, const Tensor weights, const std::vector<Tensor> gate, const std::vector<Tensor> up, const std::vector<Tensor> down, size_t intermediate, size_t num_experts) { using namespace moe_detail;
    Require(hidden.ndim()==2 && output.ndim()==2 && ids.ndim()==2 && weights.ndim()==2);
    Require(Reduced(hidden.dtype()) && output.dtype()==hidden.dtype() && ids.dtype()==DataType::kInt32 && weights.dtype()==DataType::kFloat32);
    Require(intermediate && num_experts && ids.size(1) && ids.size(1)<=num_experts && hidden.size(1) && output.shape()==hidden.shape() && weights.shape()==ids.shape() && ids.size(0)==hidden.size(0));
    Bound(hidden.size(0)); Bound(hidden.size(1)); Bound(ids.size(1)); Bound(intermediate); Bound(num_experts);
    Require(gate.size()==num_experts && up.size()==num_experts && down.size()==num_experts);
    Bound(Product(hidden.size(0),ids.size(1))); Bound(Product(Product(hidden.size(0),ids.size(1)),intermediate)); Bound(Product(Product(hidden.size(0),ids.size(1)),hidden.size(1)));
    for(size_t e=0;e<num_experts;++e) {
      Require(gate[e].ndim()==2 && up[e].ndim()==2 && down[e].ndim()==2);
      Require(gate[e].shape()==Tensor::Shape{intermediate,hidden.size(1)} && up[e].shape()==gate[e].shape() && down[e].shape()==Tensor::Shape{hidden.size(1),intermediate});
      Require(gate[e].dtype()==hidden.dtype() && up[e].dtype()==hidden.dtype() && down[e].dtype()==hidden.dtype());
    }
    Views(Collect(output,hidden,ids,weights,gate,up,down),1,true);
    // Chunked prefill consumes successive token ranges. A shifted overlap could overwrite future hidden rows.
    Require(!Overlap(output,hidden) || output.data()==hidden.data());
 }
  std::vector<Tensor> metadata_;
  size_t intermediate_, num_experts_;
};
} // namespace infini::ops
