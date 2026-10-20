#pragma once
#include "weights/weights.hpp"
#include "runtime/runtime.hpp"
#include "engine/distributed/communication_group.hpp"
#include <nlohmann/json.hpp>
namespace infinilm::models::glm_moe_dsa {
class AttentionOutput final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor local,hidden;};
 AttentionOutput(runtime::Runtime&,const nlohmann::json&,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&)const;
 size_t local_heads()const{return columns_/256;}
private:
 runtime::Runtime*owner_;size_t columns_,offset_;int tp_;infinicclComm_t comm_;bool ready_=false;
 runtime::Tensor weight_,scale_;
};
}
