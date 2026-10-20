#pragma once
#include "glm_moe_dsa_attention_prepare.hpp"
#include "glm_moe_dsa_attention_output.hpp"
#include "glm_moe_dsa_indexer_selection.hpp"
#include "layers/normalization/layer_norm.hpp"
namespace infinilm::models::glm_moe_dsa {
class NativeIndexer final:public weights::ModelWeights {
public:
 struct Features{runtime::Tensor query,key,head_weights;};
 NativeIndexer(runtime::Runtime&,const nlohmann::json&,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Features forward(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&q_resid,const runtime::Tensor&positions)const;
private:
 runtime::Runtime*owner_;size_t heads_;bool ready_=false;
 std::shared_ptr<Fp8Projection>wk_,wq_;std::shared_ptr<layers::LayerNorm>norm_;
 runtime::Tensor head_,head_float_;std::unique_ptr<layers::RoPE>rope_;
};
class NativeAttention final:public weights::ModelWeights {
public:
 using Request=IndexerSelectionR1581::Request;
 struct Output{runtime::Tensor hidden;std::vector<runtime::Tensor>selection;AttentionPrepare::Output prepared;runtime::Tensor attention;};
 NativeAttention(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows,size_t capacity,size_t topk);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&positions,const std::vector<Request>&,const std::vector<runtime::Tensor>&shared_selection={});
 void reset(runtime::Runtime&);
 const runtime::Tensor&latent_cache()const{return latent_;}
 const runtime::Tensor&rotary_cache()const{return rotary_;}
 runtime::Tensor index_cache()const{return selector_?selector_->cache():runtime::Tensor{};}
private:
 runtime::Runtime*owner_;size_t rows_,capacity_,maximum_,topk_;bool full_,ready_=false,poisoned_=false;
 std::shared_ptr<AttentionPrepare>prepare_;std::shared_ptr<AttentionOutput>output_;
 std::shared_ptr<NativeIndexer>indexer_;std::unique_ptr<IndexerSelectionR1581>selector_;
 runtime::Tensor latent_,rotary_;std::vector<size_t>lengths_;
};
}
