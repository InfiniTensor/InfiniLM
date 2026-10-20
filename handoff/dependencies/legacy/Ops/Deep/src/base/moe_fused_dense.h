#pragma once
#include "base/moe_compute_common.h"
#include <optional>
namespace infini::ops {
class MoeFusedDense : public Operator<MoeFusedDense> {
 public:

  MoeFusedDense(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1 = std::nullopt, const std::optional<Tensor> bias2 = std::nullopt) : metadata_{output, hidden, w13, w2, weights, ids, sorted, experts, total}, block_size_(block_size), bias1_(bias1), bias2_(bias2) { Validate(output, hidden, w13, w2, weights, ids, sorted, experts, total, block_size, bias1, bias2); }
  static bool supports(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1 = std::nullopt, const std::optional<Tensor> bias2 = std::nullopt) { try { Validate(output, hidden, w13, w2, weights, ids, sorted, experts, total, block_size, bias1, bias2); return true; } catch(const std::invalid_argument&) { return false; } }
  virtual void operator()(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1 = std::nullopt, const std::optional<Tensor> bias2 = std::nullopt) const = 0;
 protected:
  void Invocation(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1, const std::optional<Tensor> bias2) const {
    moe_detail::Require(block_size==block_size_); moe_detail::Same(metadata_,{output, hidden, w13, w2, weights, ids, sorted, experts, total});
    moe_detail::Require(bias1.has_value()==bias1_.has_value() && bias2.has_value()==bias2_.has_value());
    if (bias1) moe_detail::Require(std::equal_to<Tensor>{}(*bias1, *bias1_));
    if (bias2) moe_detail::Require(std::equal_to<Tensor>{}(*bias2, *bias2_));
    Validate(output, hidden, w13, w2, weights, ids, sorted, experts, total, block_size, bias1, bias2);
  }
  static void Validate(Tensor output, const Tensor hidden, const Tensor w13, const Tensor w2, const Tensor weights, const Tensor ids, const Tensor sorted, const Tensor experts, const Tensor total, size_t block_size, const std::optional<Tensor> bias1, const std::optional<Tensor> bias2) { using namespace moe_detail;
    Require(hidden.ndim()==2 && output.ndim()==2 && w13.ndim()==3 && w2.ndim()==3 && weights.ndim()==2 && ids.ndim()==2 && sorted.ndim()==1 && experts.ndim()==1 && total.ndim()==1);
    Require(Reduced(hidden.dtype()) && output.dtype()==hidden.dtype() && w13.dtype()==hidden.dtype() && w2.dtype()==hidden.dtype());
    Require(weights.dtype()==DataType::kFloat32 && ids.dtype()==DataType::kInt32 && sorted.dtype()==DataType::kInt32 && experts.dtype()==DataType::kInt32 && total.dtype()==DataType::kInt32);
    const auto n=hidden.size(0), h=hidden.size(1), e=w13.size(0), m=w2.size(2), k=ids.size(1);
    Bound(n); Bound(h); Bound(e); Bound(m); Bound(k); Bound(block_size); Bound(sorted.numel()); Bound(experts.numel());
    Require(n && h && e && m && k && block_size && output.shape()==hidden.shape());
    Require(w13.size(1)==2*m && w13.size(2)==h && w2.size(0)==e && w2.size(1)==h && weights.shape()==ids.shape() && ids.size(0)==n && total.numel()==1);
    const auto pairs=Product(n,k); Bound(pairs);
    const auto capacity=pairs<e ? Product(pairs,block_size) : pairs+Product(e,block_size-1);
    Bound(capacity); Require(sorted.numel()>=capacity && experts.numel()>=(capacity+block_size-1)/block_size);
    Bound(Product(sorted.numel(),m)); Bound(Product(sorted.numel(),h)); Bound(Product(sorted.numel(),2*m));
    std::vector<Tensor> views{output, hidden, w13, w2, weights, ids, sorted, experts, total};
    if (bias1) {
      Require(bias1->ndim()==2 && bias1->size(0)==e && bias1->size(1)==2*m && bias1->dtype()==hidden.dtype());
      views.push_back(*bias1);
    }
    if (bias2) {
      Require(bias2->ndim()==2 && bias2->size(0)==e && bias2->size(1)==h && bias2->dtype()==hidden.dtype());
      views.push_back(*bias2);
    }
    Views(views,1,true);
 }
  std::vector<Tensor> metadata_;
  size_t block_size_;
  std::optional<Tensor> bias1_, bias2_;
};
} // namespace infini::ops
