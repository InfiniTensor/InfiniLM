#include "glm_decode_leases_r1354.hpp"
#include "glm_moe_dsa_routed_ffn_r19.hpp"
#include "execution/submit.hpp"
#include <map>
#include <cmath>
#include <stdexcept>
extern "C" int glm_routed_gate_r19(void*,void*,const void*,const void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int glm_routed_down_r19(void*,const void*,const void*,const void*,const void*,const void*,size_t,size_t,void*);
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;int entered_exceptions=std::uncaught_exceptions();~Work(){decode_leases_r1354::retire(r,held,entered_exceptions);}rt::Tensor alloc(rt::Shape shape,rt::DataType type){auto x=rt::Tensor::empty(r,shape,type);held.push_back(x);return x;}rt::Tensor ids(const std::vector<int64_t>&v){auto x=alloc({v.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),v.data(),v.size()*8,false);return x;}};
}
Fp8MlpR19::Fp8MlpR19(rt::Runtime&r,size_t intermediate,const engine::distributed::RankInfo&rank):owner_(&r),local_(0),global_(intermediate),offset_(0){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(rank.tp_size!=1&&rank.tp_size!=8)||rank.tp_rank<0||rank.tp_rank>=rank.tp_size||!intermediate||intermediate%size_t(rank.tp_size)||intermediate/size_t(rank.tp_size)%128)throw std::invalid_argument("GLM MLP requires block-aligned TP1/8 partition");
 local_=intermediate/size_t(rank.tp_size);offset_=size_t(rank.tp_rank)*local_;
 try{gate_=bind_child("gate_proj",std::make_shared<Fp8Projection>(r,intermediate,6144,rank.tp_size,rank.tp_rank));up_=bind_child("up_proj",std::make_shared<Fp8Projection>(r,intermediate,6144,rank.tp_size,rank.tp_rank));down_=rt::Tensor::empty_parameter(r,{6144,local_},rt::DataType::kUInt8);scale_=rt::Tensor::empty_parameter(r,{48,global_/128},rt::DataType::kFloat32);bind_weight("down_proj.weight",weights::Weight(down_,1,rank.tp_rank,rank.tp_size));bind_weight("down_proj.weight_scale_inv",weights::Weight(scale_,-1,rank.tp_rank,rank.tp_size));r.syncStream();}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void Fp8MlpR19::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("GLM MLP postload differs");r.syncStream();ready_=true;}
rt::Tensor Fp8MlpR19::forward_local(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(&r!=owner_||!ready_||r.isGraphRecording()||!x||x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=6144||!x->is_contiguous())throw std::invalid_argument("GLM MLP requires loaded contiguous BF16 hidden");
 Work work{r};work.held.push_back(x);size_t tokens=x->size(0);auto gate=gate_->forward(r,x),up=up_->forward(r,x);work.held.insert(work.held.end(),{gate,up});auto silu=work.alloc({tokens,local_},rt::DataType::kBFloat16),activated=work.alloc({tokens,local_},rt::DataType::kBFloat16);
 // Preserve the BF16 activation boundary used by the eager checkpoint reference.
 ex::submit<infini::ops::Silu>(r,infini::ops::Silu::DefaultConfig(r.device().type()),gate,silu);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),silu,up,activated);
 auto packed=work.alloc(activated->shape(),rt::DataType::kUInt8),local_scale=work.alloc({tokens,local_/128},rt::DataType::kFloat32),global_scale=work.alloc({tokens,global_/128},rt::DataType::kFloat32),out=work.alloc({tokens,6144},rt::DataType::kBFloat16);
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),activated,packed,local_scale);r.setDeviceMemory(global_scale->data(),0xff,global_scale->nbytes());global_scale->narrow({{1,offset_/128,local_/128}})->copy_from(r,local_scale);
 ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(r.device().type()),out,packed,global_scale,down_,scale_,size_t{0},offset_);decode_leases_r1354::success_fence(r);return out;
}
FeedForwardR19::FeedForwardR19(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank):owner_(&r),sparse_(cfg.at("mlp_layer_types").at(layer)=="sparse"),tp_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(tp_!=1&&tp_!=8)||(tp_>1&&!comm_)||cfg.at("hidden_size")!=6144||cfg.at("hidden_act")!="silu"||(cfg.at("mlp_layer_types").at(layer)!="dense"&&!sparse_)||cfg.at("intermediate_size")!=12288||cfg.at("moe_intermediate_size")!=2048||cfg.at("n_routed_experts")!=256||cfg.at("num_experts_per_tok")!=8||cfg.at("n_shared_experts")!=1||cfg.at("n_group")!=1||cfg.at("topk_group")!=1||cfg.at("scoring_func")!="sigmoid"||!cfg.at("norm_topk_prob").get<bool>())throw std::invalid_argument("GLM FFN actual checkpoint configuration differs");
 try{auto alias=[&](const std::string&prefix,std::shared_ptr<Fp8MlpR19>module){for(auto&item:module->state_dict())bind_weight(prefix+item.first,item.second);return module;};
  if(!sparse_)dense_=alias("",std::make_shared<Fp8MlpR19>(r,12288,rank));
  else{gate_=rt::Tensor::empty_parameter(r,{256,6144},rt::DataType::kBFloat16);gate_float_=rt::Tensor::empty(r,{256,6144},rt::DataType::kFloat32);bias_=rt::Tensor::empty_parameter(r,{256},rt::DataType::kFloat32);bind_weight("gate.weight",weights::Weight(gate_));bind_weight("gate.e_score_correction_bias",weights::Weight(bias_));if(tp_!=8)throw std::invalid_argument("GPU-batched sparse GLM candidate requires TP8");
  routed_offset_=size_t(rank.tp_rank)*256;
  routed_gate_=rt::Tensor::empty_parameter(r,{256,256,6144},rt::DataType::kUInt8);
  routed_up_=rt::Tensor::empty_parameter(r,{256,256,6144},rt::DataType::kUInt8);
  routed_down_=rt::Tensor::empty_parameter(r,{256,6144,256},rt::DataType::kUInt8);
  routed_gate_scale_=rt::Tensor::empty_parameter(r,{256,16,48},rt::DataType::kFloat32);
  routed_up_scale_=rt::Tensor::empty_parameter(r,{256,16,48},rt::DataType::kFloat32);
  routed_down_scale_=rt::Tensor::empty_parameter(r,{256,48,16},rt::DataType::kFloat32);
  for(size_t i=0;i<256;++i){
   const auto prefix="experts."+std::to_string(i)+".";
   auto slice=[&](const rt::Tensor&t,rt::Shape shape){return t->narrow({{0,i,1}})->view(shape);};
   bind_weight(prefix+"gate_proj.weight",weights::Weight(slice(routed_gate_,{256,6144}),0,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"up_proj.weight",weights::Weight(slice(routed_up_,{256,6144}),0,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"down_proj.weight",weights::Weight(slice(routed_down_,{6144,256}),1,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"gate_proj.weight_scale_inv",weights::Weight(slice(routed_gate_scale_,{16,48}),-1,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"up_proj.weight_scale_inv",weights::Weight(slice(routed_up_scale_,{16,48}),-1,rank.tp_rank,rank.tp_size));
   bind_weight(prefix+"down_proj.weight_scale_inv",weights::Weight(slice(routed_down_scale_,{48,16}),-1,rank.tp_rank,rank.tp_size));
  }shared_=alias("shared_experts.",std::make_shared<Fp8MlpR19>(r,2048,rank));float eps=1e-20f,factor=cfg.at("routed_scaling_factor").get<float>();if(!std::isfinite(factor)||factor<=0)throw std::invalid_argument("GLM FFN invalid routed scale");epsilon_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);factor_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);r.memcpyH2D(epsilon_->data(),&eps,4,false);r.memcpyH2D(factor_->data(),&factor,4,false);}
  r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void FeedForwardR19::process_weights_after_loading(rt::Runtime&r,bool transpose){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("GLM FFN postload differs");if(dense_)dense_->process_weights_after_loading(r,transpose);else{for(auto&expert:experts_)expert->process_weights_after_loading(r,transpose);shared_->process_weights_after_loading(r,transpose);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),gate_,gate_float_);}r.syncStream();ready_=true;}
FeedForwardR19::Output FeedForwardR19::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(&r!=owner_||!ready_||r.isGraphRecording()||!x||x->device()!=r.device()||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->is_contiguous()||!x->size(0)||x->size(0)>4096||x->size(1)!=6144)throw std::invalid_argument("GLM FFN requires loaded BF16 packed hidden");
 Work work{r};work.held.push_back(x);size_t tokens=x->size(0);auto combined=work.alloc({tokens,6144},rt::DataType::kFloat32);rt::Tensor indices,weights;
 if(dense_){auto local=dense_->forward_local(r,x);work.held.push_back(local);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),local,combined);}
 else{auto input=work.alloc(x->shape(),rt::DataType::kFloat32),logits=work.alloc({tokens,256},rt::DataType::kFloat32),scores=work.alloc({tokens,256},rt::DataType::kFloat32),choice=work.alloc({tokens,256},rt::DataType::kFloat32),chosen=work.alloc({tokens,8},rt::DataType::kFloat32);indices=work.alloc({tokens,8},rt::DataType::kInt64);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),x,input);ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),input,gate_float_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},logits);ex::submit<infini::ops::Sigmoid>(r,infini::ops::Sigmoid::DefaultConfig(r.device().type()),logits,scores);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),scores,bias_,choice);ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(r.device().type()),choice,int64_t{8},int64_t{1},true,true,chosen,indices);
  auto selected=work.alloc({tokens,8},rt::DataType::kFloat32),sum=work.alloc({tokens,1},rt::DataType::kFloat32),denominator=work.alloc({tokens,1},rt::DataType::kFloat32),normalized=work.alloc({tokens,8},rt::DataType::kFloat32);weights=work.alloc({tokens,8},rt::DataType::kFloat32);
  ex::submit<infini::ops::Gather>(r,infini::ops::Gather::DefaultConfig(r.device().type()),scores,indices,int64_t{1},false,selected);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),selected,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},sum);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),sum,epsilon_,denominator);ex::submit<infini::ops::Div>(r,infini::ops::Div::DefaultConfig(r.device().type()),selected,denominator,normalized);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),normalized,factor_,weights);
  // indices are produced immediately above by public Topk over 256 experts;
  // their range is established by that operator, not unchecked user metadata.
  // All banks alias the original checkpoint names; no duplicate expert weights.
  const size_t routes=tokens*8;
  auto packed=work.alloc({tokens,6144},rt::DataType::kUInt8),input_scale=work.alloc({tokens,48},rt::DataType::kFloat32);
  ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),x,packed,input_scale);
  auto gates=work.alloc({routes,256},rt::DataType::kBFloat16),ups=work.alloc({routes,256},rt::DataType::kBFloat16);
  if(glm_routed_gate_r19(gates->data(),ups->data(),packed->data(),input_scale->data(),indices->data(),routed_gate_->data(),routed_gate_scale_->data(),routed_up_->data(),routed_up_scale_->data(),routes,routed_offset_,r.stream()))throw std::runtime_error("GLM routed gate/up launch failed");
  auto silu=work.alloc({routes,256},rt::DataType::kBFloat16),activated=work.alloc({routes,256},rt::DataType::kBFloat16);
  ex::submit<infini::ops::Silu>(r,infini::ops::Silu::DefaultConfig(r.device().type()),gates,silu);
  ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),silu,ups,activated);
  auto down_input=work.alloc({routes,256},rt::DataType::kUInt8),down_scale=work.alloc({routes,2},rt::DataType::kFloat32),local=work.alloc({routes,6144},rt::DataType::kBFloat16);
  ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(r.device().type()),activated,down_input,down_scale);
  if(glm_routed_down_r19(local->data(),down_input->data(),down_scale->data(),indices->data(),routed_down_->data(),routed_down_scale_->data(),routes,routed_offset_,r.stream()))throw std::runtime_error("GLM routed down launch failed");
  auto local_float=work.alloc({routes,6144},rt::DataType::kFloat32),contributions=work.alloc({routes,6144},rt::DataType::kFloat32);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),local,local_float);
  ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),local_float,weights->view({routes,1}),contributions);
  auto routed=work.alloc({tokens,6144},rt::DataType::kFloat32),shared=shared_->forward_local(r,x),shared_float=work.alloc({tokens,6144},rt::DataType::kFloat32);work.held.push_back(shared);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),contributions->view({tokens,8,6144}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},routed);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),shared,shared_float);ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(r.device().type()),routed,shared_float,combined);
 }
 auto reduced=combined;if(tp_>1){reduced=work.alloc(combined->shape(),combined->dtype());reduced->copy_from(r,combined);engine::distributed::allreduce_(r,reduced,reduced,infinicclSum,comm_);}auto out=work.alloc({tokens,6144},rt::DataType::kBFloat16);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),reduced,out);decode_leases_r1354::success_fence(r);return {out,combined,weights,indices};
}
}
