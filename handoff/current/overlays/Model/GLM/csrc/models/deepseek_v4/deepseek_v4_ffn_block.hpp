#pragma once
#include "deepseek_v4_moe_eager.hpp"
#include "deepseek_v4_hyper_connection.hpp"
#include "layers/normalization/rmsnorm.hpp"
namespace infinilm::models::deepseek_v4 {
class DeepseekV4FFNBlock final:public weights::ModelWeights {
public:
 struct Prepared{HyperConnection::PreResult hc;runtime::Tensor normalized;};
 DeepseekV4FFNBlock(runtime::Runtime &,size_t intermediate,size_t shared,size_t vocab,bool hash,double factor,
                   const engine::distributed::RankInfo &,double rms_eps,double hc_eps,int iterations);
 void process_weights_after_loading(runtime::Runtime &,bool)override;
 Prepared prepare(runtime::Runtime &,const runtime::Tensor &)const;
 DeepseekV4MoEEager::Output forward(runtime::Runtime &,const runtime::Tensor &residual,const runtime::Tensor &ids)const;
private:
 std::shared_ptr<HyperConnection>hc_;std::shared_ptr<layers::RMSNorm>norm_;std::shared_ptr<DeepseekV4MoEEager>moe_;bool ready_=false;
};
}
