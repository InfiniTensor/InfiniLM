#include "glm_moe_dsa_attention_output.hpp"
#include "execution/submit.hpp"
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape shape,rt::DataType type){auto t=rt::Tensor::empty(r,shape,type);held.push_back(t);return t;}};}
AttentionOutput::AttentionOutput(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):owner_(&r),columns_(0),offset_(0),tp_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(tp_!=1&&tp_!=8)||rank.tp_rank<0||rank.tp_rank>=tp_||(tp_>1&&!comm_)||cfg.at("hidden_size")!=6144||cfg.at("num_attention_heads")!=64||cfg.at("v_head_dim")!=256||cfg.value("attention_bias",false))throw std::invalid_argument("GLM attention output geometry/owner/TP differs");
 columns_=16384/size_t(tp_);offset_=size_t(rank.tp_rank)*columns_;
 try{weight_=rt::Tensor::empty_parameter(r,{6144,columns_},rt::DataType::kUInt8);scale_=rt::Tensor::empty_parameter(r,{48,128},rt::DataType::kFloat32);bind_weight("o_proj.weight",weights::Weight(weight_,1,rank.tp_rank,tp_));bind_weight("o_proj.weight_scale_inv",weights::Weight(scale_,-1,rank.tp_rank,tp_));}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void AttentionOutput::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(owner_!=&r||ready_||r.isGraphRecording())throw std::logic_error("GLM output postload owner/state differs");r.syncStream();ready_=true;}
AttentionOutput::Output AttentionOutput::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(owner_!=&r||!ready_||r.isGraphRecording()||!x||x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=3||!x->size(0)||x->size(1)!=local_heads()||x->size(2)!=256||!x->is_contiguous())throw std::invalid_argument("GLM attention output requires contiguous local BF16 V256 heads");
 Work work{r};size_t tokens=x->size(0);auto input=x->view({tokens,columns_});auto packed=work.alloc(input->shape(),rt::DataType::kUInt8),local_scale=work.alloc({tokens,columns_/128},rt::DataType::kFloat32),global_scale=work.alloc({tokens,128},rt::DataType::kFloat32),local=work.alloc({tokens,6144},rt::DataType::kBFloat16);
 // Each TP shard starts and ends on a 128-element activation block. Local
 // quantization is therefore identical to quantizing before the split. Keep
 // scale indexing in the original global blocks used by the checkpoint.
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),input,packed,local_scale);
 r.setDeviceMemory(global_scale->data(),0xff,global_scale->nbytes());global_scale->narrow({{1,offset_/128,columns_/128}})->copy_from(r,local_scale);
 ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()),local,packed,global_scale,weight_,scale_,size_t{0},offset_);
 auto hidden=local;if(tp_>1){hidden=work.alloc(local->shape(),local->dtype());hidden->copy_from(r,local);engine::distributed::allreduce_(r,hidden,hidden,infinicclSum,comm_);}r.syncStream();return {local,hidden};
}
}
