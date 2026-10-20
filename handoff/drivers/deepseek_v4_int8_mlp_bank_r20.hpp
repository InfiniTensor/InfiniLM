#pragma once
#include "weights/weights.hpp"
#include "engine/distributed/communication_group.hpp"

namespace infinilm::models::deepseek_v4 {
// Original checkpoint layout, no NVIDIA Marlin repack. The MoE owner combines
// this TP-local contribution with routed experts before its single allreduce.
class DeepseekV4Int8MLPBankR20 : public weights::ModelWeights {
 public:
  DeepseekV4Int8MLPBankR20(runtime::Runtime &,size_t hidden,size_t intermediate,
                   const runtime::DataType &,const engine::distributed::RankInfo &,double swiglu_limit=10.0);
  DeepseekV4Int8MLPBankR20(runtime::Runtime &,size_t hidden,size_t intermediate,
      const runtime::DataType &,const engine::distributed::RankInfo &,
      runtime::Tensor w13,runtime::Tensor s13,runtime::Tensor w2,runtime::Tensor s2);
  void process_weights_after_loading(runtime::Runtime &,bool) override;
  runtime::Tensor forward_local(runtime::Runtime &,const runtime::Tensor &) const;
 private:
  runtime::Device device_;
  runtime::DataType dtype_;
  double swiglu_limit_;
  size_t hidden_,local_intermediate_;
  runtime::Tensor w13_,s13_,w2_,s2_;
  bool ready_{false};
  bool postload_attempted_{false};
};
}
