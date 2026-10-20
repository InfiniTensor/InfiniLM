#include "glm_prefix_proof_r1645.hpp"
#include <cstdint>
#include <stdexcept>
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
 std::vector<runtime::Tensor> select_all_metadata(runtime::Runtime&,const std::vector<Request>&);
 // Explicit private read-only borrowing API. Consumers must not write these
 // immutable index views; ordinary select_all_metadata remains unchanged.
 struct OwnedPrefixStats {uint64_t bank_builds=0,calls=0,returned_views=0;};
 std::vector<runtime::Tensor> select_all_metadata_borrowed_readonly(runtime::Runtime&,const std::vector<Request>&);
 OwnedPrefixStats owned_prefix_stats(runtime::Runtime&r)const {r.assertOwnerThread();if(&r!=owner_)throw std::logic_error("1581 foreign stats Runtime");return prefix_stats_r1581_;}
 ProvenPrefixR1645 select_all_metadata_proven_readonly(runtime::Runtime&,const std::vector<Request>&);
 runtime::Tensor cache()const{return cache_;}
 void reset(runtime::Runtime &);
private:
 runtime::Tensor prefix_bank_r1581_;OwnedPrefixStats prefix_stats_r1581_;
 runtime::Runtime*owner_;std::vector<bool>missing_keys_r1406_;
 runtime::Device device_;size_t rows_,capacity_,heads_,local_heads_,dim_,topk_;bool poisoned_=false;
 runtime::Tensor cache_,head_scale_;std::vector<size_t> lengths_;int tp_size_;infinicclComm_t comm_;
};
}
