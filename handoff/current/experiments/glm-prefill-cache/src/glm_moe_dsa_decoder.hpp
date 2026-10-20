#pragma once
#include "glm_moe_dsa_native_attention.hpp"
#include "glm_moe_dsa_ffn.hpp"
namespace infinilm::models::glm_moe_dsa {
class Decoder final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor hidden,normalized_attention,residual,normalized_ffn;NativeAttention::Output attention;FeedForward::Output ffn;};
 Decoder(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows,size_t capacity,size_t topk);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<NativeAttention::Request>&,const std::vector<runtime::Tensor>&shared={});
 void reset(runtime::Runtime&);
 std::vector<runtime::Tensor>cache_tensors()const{return {attention_->latent_cache(),attention_->rotary_cache(),attention_->index_cache()};}
private:
 runtime::Runtime*owner_;float eps_;bool ready_=false,poisoned_=false;
 runtime::Tensor input_norm_,post_norm_,ones_;
 std::unique_ptr<NativeAttention>attention_;std::unique_ptr<FeedForward>ffn_;
};
}
