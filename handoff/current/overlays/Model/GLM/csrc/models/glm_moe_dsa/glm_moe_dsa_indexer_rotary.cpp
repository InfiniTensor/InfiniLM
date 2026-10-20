#include "glm_moe_dsa_indexer_rotary.hpp"
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
std::pair<runtime::Tensor,runtime::Tensor> indexer_rotary(
 runtime::Runtime &runtime,const layers::RoPE &rope,
 const runtime::Tensor &q,const runtime::Tensor &k,
 const runtime::Tensor &positions,size_t rotary_dim){
 runtime.assertOwnerThread();
 if(!q||!k||q->ndim()!=3||k->ndim()!=3||k->size(1)!=1||!q->size(0)||!q->size(1)
    ||q->size(0)!=k->size(0)||q->size(2)!=k->size(2)||!rotary_dim||rotary_dim%2||rotary_dim>q->size(2)
    ||q->dtype()!=k->dtype()||q->device()!=runtime.device()||k->device()!=runtime.device())
  throw std::invalid_argument("GLM indexer rotary requires same-owner [T,H,D] query and [T,1,D] key with even rotary prefix");
 auto rotated=rope.forward(runtime,q,k,positions);
 auto pack=[&](const runtime::Tensor &x){
  auto out=runtime::Tensor::empty(runtime,x->shape(),x->dtype());out->copy_from(runtime,x);
  auto prefix=x->narrow({{2,0,rotary_dim}})->view({x->size(0),x->size(1),rotary_dim/2,2});
  auto grouped=prefix->permute({0,1,3,2})->contiguous(runtime)->view({x->size(0),x->size(1),rotary_dim});
  out->narrow({{2,0,rotary_dim}})->copy_from(runtime,grouped);return out;
 };
 return {pack(rotated.first),pack(rotated.second)};
}
}
