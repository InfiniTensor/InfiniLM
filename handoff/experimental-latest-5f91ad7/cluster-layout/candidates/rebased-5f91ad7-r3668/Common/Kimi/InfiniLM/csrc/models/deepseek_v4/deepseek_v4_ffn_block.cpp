#include "deepseek_v4_ffn_block.hpp"
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
DeepseekV4FFNBlock::DeepseekV4FFNBlock(runtime::Runtime&r,size_t intermediate,size_t shared,size_t vocab,bool hash,double factor,const engine::distributed::RankInfo&rank,double rms,double eps,int iterations){
 hc_=std::make_shared<HyperConnection>(r,rms,eps,iterations);
 // Keep the checkpoint's flat hc_ffn_{fn,base,scale} names. The owned HC
 // module's hook is invoked here; it is deliberately not a second weight child.
 for(auto&entry:hc_->state_dict())bind_weight("hc_ffn_"+entry.first,entry.second);
 norm_=bind_child("ffn_norm",std::make_shared<layers::RMSNorm>(r,4096,rms,runtime::DataType::kBFloat16));
 moe_=bind_child("ffn",std::make_shared<DeepseekV4MoEEager>(r,4096,intermediate,shared,256,6,vocab,hash,factor,rank));
}
void DeepseekV4FFNBlock::process_weights_after_loading(runtime::Runtime&r,bool transpose){if(ready_)throw std::logic_error("DeepSeek FFN block already finalized");hc_->process_weights_after_loading(r,transpose);r.syncStream();ready_=true;}
DeepseekV4MoEEager::Output DeepseekV4FFNBlock::forward(runtime::Runtime&r,const runtime::Tensor&residual,const runtime::Tensor&ids)const{
 auto prepared=prepare(r,residual);auto moe=moe_->forward(r,prepared.normalized,ids);auto value=hc_->post(r,moe.value,residual,prepared.hc);return {value,moe.weights,moe.indices};
}
DeepseekV4FFNBlock::Prepared DeepseekV4FFNBlock::prepare(runtime::Runtime&r,const runtime::Tensor&residual)const{
 if(!ready_)throw std::logic_error("DeepSeek FFN block is not loaded");auto pre=hc_->pre(r,residual);auto normalized=norm_->forward(r,pre.collapsed);r.syncStream();return {pre,normalized};
}
}
