#pragma once
#include "glm_moe_dsa_attention_prepare.hpp"
namespace infinilm::models::glm_moe_dsa {
class Fp8Mlp final:public weights::ModelWeights {
public:
 Fp8Mlp(runtime::Runtime&,size_t intermediate,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 runtime::Tensor forward_local(runtime::Runtime&,const runtime::Tensor&)const;
private:
 runtime::Runtime*owner_;size_t local_,global_,offset_;bool ready_=false;
 std::shared_ptr<Fp8Projection>gate_,up_;runtime::Tensor down_,scale_;
};
class FeedForward final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor hidden,local,route_weights,route_indices;};
 FeedForward(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&)const;
 bool sparse()const{return sparse_;}
private:
 runtime::Runtime*owner_;bool sparse_,ready_=false;int tp_;infinicclComm_t comm_;
 std::shared_ptr<Fp8Mlp>dense_,shared_;std::vector<std::shared_ptr<Fp8Mlp>>experts_;
 runtime::Tensor gate_,gate_float_,bias_,epsilon_,factor_;
};
}
