#include "deepseek_v4_attention_output.hpp"
#include "execution/submit.hpp"
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}
AttentionOutput::AttentionOutput(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):device_(r.device()),groups_(0),tp_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||(tp_!=1&&tp_!=8)||rank.tp_rank<0||rank.tp_rank>=tp_||(tp_>1&&!comm_)||cfg.at("hidden_size")!=4096||cfg.at("head_dim")!=512||cfg.at("num_attention_heads")!=64||cfg.at("o_groups")!=8||cfg.at("o_lora_rank")!=1024)throw std::invalid_argument("DeepSeek output checkpoint geometry/owner/TP differs");
 groups_=8/size_t(tp_);
 try{
 a_=rt::Tensor::empty_parameter(r,{groups_*1024,4096},rt::DataType::kBFloat16);
 b_=rt::Tensor::empty_parameter(r,{4096,groups_*1024},rt::DataType::kInt8);
 bs_=rt::Tensor::empty_parameter(r,{4096,1},rt::DataType::kFloat32);
 bind_weight("wo_a.weight",weights::Weight(a_,0,rank.tp_rank,tp_));
 bind_weight("wo_b.weight",weights::Weight(b_,1,rank.tp_rank,tp_));
 bind_weight("wo_b.scale",weights::Weight(bs_,-1,rank.tp_rank,tp_));
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void AttentionOutput::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("DeepSeek output postload state differs");r.syncStream();ready_=true;}
AttentionOutput::Output AttentionOutput::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();
 if(!ready_||r.device()!=device_||r.isGraphRecording()||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||!x->is_contiguous()||x->ndim()!=3||!x->size(0)||x->size(1)!=groups_*8||x->size(2)!=512)throw std::invalid_argument("DeepSeek output requires local contiguous inverse-rotated attention heads");
 Work work{r};size_t tokens=x->size(0);
 auto grouped=work.alloc({groups_,tokens,1024},rt::DataType::kBFloat16);
 ex::submit<infini::ops::Gemm>(r,ex::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(device_.type()),x->view({tokens,groups_,4096})->permute({1,0,2}),a_->view({groups_,1024,4096})->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1.f},std::optional<float>{0.f},std::optional<int>{},std::optional<int>{},grouped);
 auto flat=grouped->permute({1,0,2})->contiguous(r)->view({tokens,groups_*1024});work.held.push_back(flat);
 auto packed=work.alloc(flat->shape(),rt::DataType::kInt8),scale=work.alloc({tokens,1},rt::DataType::kFloat32),local=work.alloc({tokens,4096},rt::DataType::kBFloat16);
 auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(device_.type());quant.set_implementation_index(0);
 auto gemm=infini::ops::I8Gemm::DefaultConfig(device_.type());gemm.set_implementation_index(0);
 ex::submit<infini::ops::PerChannelQuantI8>(r,ex::RecordingPolicy::HostOnly,quant,flat,packed,scale);
 ex::submit<infini::ops::I8Gemm>(r,ex::RecordingPolicy::HostOnly,gemm,local,packed,scale,b_->permute({1,0}),bs_,std::optional<rt::Tensor>{});
 auto hidden=local;
 if(tp_>1){hidden=work.alloc(local->shape(),local->dtype());hidden->copy_from(r,local);engine::distributed::allreduce_(r,hidden,hidden,infinicclSum,comm_);}
 r.syncStream();return {flat,local,hidden};
}
}
