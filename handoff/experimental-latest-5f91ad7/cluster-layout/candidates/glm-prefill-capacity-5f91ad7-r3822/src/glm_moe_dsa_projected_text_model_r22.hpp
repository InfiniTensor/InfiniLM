#pragma once
#include "glm_moe_dsa_projected_decoder_r22.hpp"

namespace infinilm::models::glm_moe_dsa {
// Complete checkpoint text stack. The ordinary Engine wrapper must supply
// explicit request rows and own the borrowed TP/PP communicators.
class TextModelProjectedR22 final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor hidden,logits;};
 TextModelProjectedR22(runtime::Runtime&,const nlohmann::json&,
           const engine::distributed::RankInfo&,size_t rows,size_t capacity,const bool*bounded_policy=nullptr,const bool*owned_prefix_policy=nullptr);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&token_ids,
                const runtime::Tensor&positions,
                const std::vector<NativeAttentionProjectedR22::Request>&,
                bool all_positions=false);
 void reset(runtime::Runtime&);
 size_t layer_begin()const{return begin_;}
 size_t layer_end()const{return end_;}
 bool all_sequences_empty(runtime::Runtime&r)const{check_owner(r);for(auto length:lengths_)if(length)return false;return true;}
 bool first_stage()const{return rank_.pp_stage==0;}
 bool last_stage()const{return rank_.pp_stage+1==rank_.pp_size;}
private:
 void check_owner(runtime::Runtime&)const;
 runtime::Runtime*owner_;engine::distributed::RankInfo rank_;
 nlohmann::json cfg_;size_t rows_,capacity_,vocab_,begin_,end_;
 float eps_;bool ready_=false,poisoned_=false;
 runtime::Tensor embedding_,norm_,ones_,head_;
 std::vector<std::unique_ptr<DecoderProjectedR22>>layers_;
 std::vector<size_t>lengths_;
};
}
