#pragma once
#include "glm_moe_dsa_native_attention.hpp"
namespace infinilm::models::glm_moe_dsa {
class NativeAttentionProjectedR22 final:public weights::ModelWeights {
public:
 using Request=IndexerSelection::Request;
 struct Output{runtime::Tensor hidden;std::vector<runtime::Tensor>selection;AttentionPrepare::Output prepared;runtime::Tensor attention;};
 NativeAttentionProjectedR22(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows,size_t capacity,size_t topk);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&positions,const std::vector<Request>&,const std::vector<runtime::Tensor>&shared_selection={});
 void reset(runtime::Runtime&);
 const runtime::Tensor&latent_cache()const{return latent_;}
 const runtime::Tensor&rotary_cache()const{return rotary_;}
 runtime::Tensor index_cache()const{return selector_?selector_->cache():runtime::Tensor{};}
 const runtime::Tensor&projected_key_cache()const{return projected_key_;}
 const runtime::Tensor&projected_value_cache()const{return projected_value_;}
private:
 runtime::Runtime*owner_;size_t rows_,capacity_,maximum_,topk_;bool full_,ready_=false,poisoned_=false;
 std::shared_ptr<AttentionPrepare>prepare_;std::shared_ptr<AttentionOutput>output_;
 std::shared_ptr<NativeIndexer>indexer_;std::unique_ptr<IndexerSelection>selector_;
 runtime::Tensor latent_,rotary_,projected_key_,projected_value_;std::vector<size_t>lengths_;
};
}
