#pragma once
#include "deepseek_v4_attention_prepare.hpp"
#include "deepseek_v4_attention_output.hpp"
#include "deepseek_v4_c4_indexer.hpp"
namespace infinilm::models::deepseek_v4 {
// C4 CSA: causal sliding window plus global-head indexer-selected compressed groups.
class CsaAttention final:public weights::ModelWeights {
public:
 struct Request{size_t row,past,tokens;};
 CsaAttention(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 runtime::Tensor forward(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<Request>&);
 void reset(runtime::Runtime&);
 const runtime::Tensor&raw_cache()const{return cache_->raw();}
 runtime::Tensor raw_compressed_cache()const{return compressor_->raw_cache();}
 const runtime::Tensor&compressor_state()const{return compressor_->state();}
 const std::vector<runtime::Tensor>&last_selection()const{return last_selection_;}
 runtime::Tensor raw_index_cache()const{return indexer_->raw_cache();}
 runtime::Tensor indexer_state()const{return indexer_->state();}
private:
 runtime::Runtime*owner_;size_t rows_,window_,maximum_,compressed_capacity_;bool ready_=false,poisoned_=false;
 std::shared_ptr<AttentionPrepare>prepare_;std::shared_ptr<AttentionOutput>output_;
 std::unique_ptr<PagedLatentCache>cache_;std::unique_ptr<TemporalCompressor>compressor_;std::unique_ptr<C4Indexer>indexer_;
 size_t topk_;std::vector<runtime::Tensor>last_selection_;
 runtime::Tensor sink_;std::vector<size_t>lengths_;
};
}
