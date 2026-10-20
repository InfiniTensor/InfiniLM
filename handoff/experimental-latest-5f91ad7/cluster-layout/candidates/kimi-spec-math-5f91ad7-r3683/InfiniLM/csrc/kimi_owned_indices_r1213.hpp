#pragma once
#include "runtime/tensor.hpp"
#include <array>
#include <memory>
#include <vector>
namespace infinilm::models::kimi_native {
namespace rt=infinilm::runtime;
// Internal eager decode scope. No default SDK/provider or graph eligibility change.
class OwnedIndicesRequestR1213 final {
public:
 OwnedIndicesRequestR1213(rt::Runtime&runtime,const rt::Tensor&offsets,const rt::Tensor&reads,const rt::Tensor&writes,const std::vector<rt::Tensor>&conv_pools,const std::vector<rt::Tensor>&ssm_pools,int mode);
 ~OwnedIndicesRequestR1213()noexcept;
 OwnedIndicesRequestR1213(const OwnedIndicesRequestR1213&)=delete;
 OwnedIndicesRequestR1213&operator=(const OwnedIndicesRequestR1213&)=delete;
 void finish();
 bool matches(rt::Runtime&runtime)const;
 void kda(const std::array<rt::Tensor,9>&values,float scale,float lower,bool normalize);
 void conv(const std::array<rt::Tensor,4>&values);
private:
 rt::Runtime*runtime_;std::shared_ptr<rt::Memory>metadata_,workspace_;void*raw_{};
};
bool owned_indices_active_r1213(rt::Runtime&runtime);
void owned_indices_kda_r1213(rt::Runtime&runtime,const rt::Tensor&out,const rt::Tensor&state,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&gate,const rt::Tensor&beta,const rt::Tensor&alog,const rt::Tensor&bias,float scale,float lower,bool normalize);
void owned_indices_conv_r1213(rt::Runtime&runtime,const rt::Tensor&out,const rt::Tensor&state,const rt::Tensor&qkv,const rt::Tensor&weight);
}
