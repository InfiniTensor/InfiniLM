#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace infinilm::models::glm_moe_dsa {
// The checkpoint stores projections only on full indexer layers. Shared layers
// consume the nearest earlier full layer's top-k result for the same forward.
// This metadata plan does not own/cache device results across requests.
class IndexerPlan {
 public:
  explicit IndexerPlan(const std::vector<bool> &full):full_(full){
    if(!full.empty()&&!full.front())throw std::invalid_argument("GLM first indexer layer must own weights");
    size_t source=0;
    for(size_t i=0;i<full.size();++i){if(full[i])source=i;sources_.push_back(source);}
  }
  bool owns_weights(size_t layer) const {return full_.at(layer);}
  size_t source_layer(size_t layer) const {return sources_.at(layer);}
  void validate_partition(size_t begin,size_t end) const {
    if(begin>end||end>full_.size())throw std::invalid_argument("GLM indexer partition bounds invalid");
    if(begin==end)return;
    // A PP boundary at a shared layer needs explicit cross-stage index payload.
    // The current migration keeps that path closed and uses full-layer boundaries.
    if(!full_[begin])throw std::invalid_argument("GLM PP partition begins at shared indexer without payload");
    for(size_t i=begin;i<end;++i)if(sources_[i]<begin)throw std::invalid_argument("GLM indexer source crosses PP owner");
  }
 private:
  std::vector<bool> full_;
  std::vector<size_t> sources_;
};
}
