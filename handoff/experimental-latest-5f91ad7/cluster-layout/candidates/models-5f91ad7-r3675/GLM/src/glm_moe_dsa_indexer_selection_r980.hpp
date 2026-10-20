#pragma once
#include "runtime/runtime.hpp"
#include "engine/distributed/communication_group.hpp"
#include <vector>
namespace infinilm::models::glm_moe_dsa {
// Eager correctness baseline. Projections, normalization and rotary are supplied
// by the model. Cache rows follow request identities, independently of batch order.
class IndexerSelectionR1581 {
public:
 struct Request{size_t row,past,tokens;};
 struct Result{runtime::Tensor scores,indices;};
 IndexerSelectionR1581(runtime::Runtime &,size_t rows,size_t capacity,size_t heads,size_t dim,size_t topk,
                  const engine::distributed::RankInfo &);
 std::vector<Result> forward(runtime::Runtime &,const runtime::Tensor &query,const runtime::Tensor &key,
                             const runtime::Tensor &head_weights,const std::vector<Request>&);
 // Decode-only all-visible-keys path. No scores are fabricated; forward() keeps
 // its original public scoring contract. Returned index ranges do not overlap.
 std::vector<runtime::Tensor> select_all(runtime::Runtime&,const runtime::Tensor&key,const std::vector<Request>&);
 runtime::Tensor cache()const{return cache_;}
 void reset(runtime::Runtime &);
private:
 runtime::Device device_;size_t rows_,capacity_,heads_,local_heads_,dim_,topk_;bool poisoned_=false;
 runtime::Tensor cache_,head_scale_;std::vector<size_t> lengths_;int tp_size_;infinicclComm_t comm_;
};
}
