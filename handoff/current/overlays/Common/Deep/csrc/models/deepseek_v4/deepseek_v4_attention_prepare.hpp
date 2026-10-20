#pragma once
#include "weights/weights.hpp"
#include "runtime/runtime.hpp"
#include "layers/normalization/rmsnorm.hpp"
#include "engine/distributed/communication_group.hpp"
#include <nlohmann/json.hpp>
namespace infinilm::models::deepseek_v4 {
class AttentionPrepare final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor query,kv,q_lora;};
 AttentionPrepare(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&positions)const;
 runtime::Tensor inverse_output_rope(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&positions)const;
 size_t local_heads()const{return local_heads_;}
private:
 void rotate(runtime::Runtime&,runtime::Tensor,std::optional<runtime::Tensor>,const runtime::Tensor&,bool)const;
 runtime::Device device_;size_t local_heads_,max_position_;double eps_,theta_,factor_,beta_fast_,beta_slow_;size_t original_;bool ready_=false;
 runtime::Tensor qa_,qas_,kv_,kvs_,qb_,qbs_,ones_;
 std::shared_ptr<layers::RMSNorm>q_norm_,kv_norm_;
};
}
