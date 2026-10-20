#pragma once
#include "deepseek_v4_attention_prepare.hpp"
#include "deepseek_v4_attention_output.hpp"
#include "deepseek_v4_paged_latent_cache.hpp"
namespace infinilm::models::deepseek_v4 {
class SwaAttention final:public weights::ModelWeights {
public:
 struct Request{size_t row,past,tokens;};
 SwaAttention(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 runtime::Tensor forward(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&positions,const std::vector<Request>&);
 void reset(runtime::Runtime&);
 const runtime::Tensor&raw_cache()const{return cache_->raw();}
private:
 runtime::Runtime*owner_;size_t rows_,window_,maximum_;bool ready_=false;
 std::shared_ptr<AttentionPrepare>prepare_;std::shared_ptr<AttentionOutput>output_;std::unique_ptr<PagedLatentCache>cache_;runtime::Tensor sink_;std::vector<size_t>lengths_;
};
}
