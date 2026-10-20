#pragma once
#include "runtime/runtime.hpp"
#include "weights/weights.hpp"
namespace infinilm::models::deepseek_v4 {
// Public-operator correctness baseline matching the original checkpoint's
// FP32 pre/post/comb contract, including comb[source,destination] orientation.
class HyperConnection final:public weights::ModelWeights {
public:
 struct PreResult{runtime::Tensor collapsed,post,comb;};
 HyperConnection(runtime::Runtime &,double rms_eps,double hc_eps,int sinkhorn_iterations);
 void process_weights_after_loading(runtime::Runtime &,bool)override;
 PreResult pre(runtime::Runtime &,const runtime::Tensor &)const;
 runtime::Tensor post(runtime::Runtime &,const runtime::Tensor &block_output,const runtime::Tensor &residual,const PreResult &)const;
private:
 runtime::Device device_;double rms_eps_,hc_eps_;int iterations_;bool ready_=false;
 runtime::Tensor fn_,base_,scale_,constants_;
};
}
