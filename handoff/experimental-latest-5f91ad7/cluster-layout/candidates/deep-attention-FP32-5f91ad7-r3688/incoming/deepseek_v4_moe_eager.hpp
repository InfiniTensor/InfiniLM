#pragma once
#include "deepseek_v4_int8_mlp.hpp"
#include "runtime/runtime.hpp"
#include <vector>
namespace infinilm::models::deepseek_v4 {
// Correctness baseline on the new Runtime: raw checkpoint INT8 experts,
// explicit TP ownership, FP32 routed/shared accumulation and one TP reduction.
// Route metadata is copied to the host; this path is eager and not graph eligible.
class DeepseekV4MoEEager final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor value,weights,indices;};
 DeepseekV4MoEEager(runtime::Runtime &,size_t hidden,size_t intermediate,size_t shared_intermediate,
  size_t experts,size_t topk,size_t vocab,bool hash,double factor,const engine::distributed::RankInfo &);
 void process_weights_after_loading(runtime::Runtime &,bool) override;
 Output forward(runtime::Runtime &,const runtime::Tensor &,const runtime::Tensor &) const;
private:
 runtime::Device device_;size_t hidden_,experts_,topk_,vocab_;bool hash_,ready_=false;int tp_size_;infinicclComm_t comm_;
 runtime::Tensor gate_,bias_,table_,epsilon_,factor_;
 std::vector<std::shared_ptr<DeepseekV4Int8MLP>> experts_modules_;
 std::shared_ptr<DeepseekV4Int8MLP> shared_;
};
}
