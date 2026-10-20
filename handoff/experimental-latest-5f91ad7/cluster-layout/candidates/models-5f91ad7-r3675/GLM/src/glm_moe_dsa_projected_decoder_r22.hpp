#pragma once
#include "glm_moe_dsa_projected_cache_r22.hpp"
#include "glm_moe_dsa_routed_ffn_r19.hpp"
namespace infinilm::models::glm_moe_dsa {
class DecoderProjectedR22 final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor hidden,normalized_attention,residual,normalized_ffn;NativeAttentionProjectedR22::Output attention;FeedForwardR19::Output ffn;};
 DecoderProjectedR22(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows,size_t capacity,size_t topk,const bool*bounded_policy=nullptr,const bool*owned_prefix_policy=nullptr);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<NativeAttentionProjectedR22::Request>&,const std::vector<runtime::Tensor>&shared={},const PrefixProofsR1645&shared_proofs={});
 void reset(runtime::Runtime&);
 std::vector<runtime::Tensor>cache_tensors()const{return {attention_->latent_cache(),attention_->rotary_cache(),attention_->index_cache(),attention_->projected_key_cache(),attention_->projected_value_cache()};}
private:
 runtime::Runtime*owner_;float eps_;bool ready_=false,poisoned_=false;
 runtime::Tensor input_norm_,post_norm_,ones_;
 std::unique_ptr<NativeAttentionProjectedR22>attention_;std::unique_ptr<FeedForwardR19>ffn_;
};
}
