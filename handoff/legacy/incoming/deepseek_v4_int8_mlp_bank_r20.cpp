#include "deepseek_v4_int8_mlp_bank_r20.hpp"
#include "execution/submit.hpp"
#include <exception>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
DeepseekV4Int8MLPBankR20::DeepseekV4Int8MLPBankR20(runtime::Runtime &rt,size_t hidden,size_t intermediate,
    const runtime::DataType &dtype,const engine::distributed::RankInfo &rank,double swiglu_limit)
    :device_(rt.device()),dtype_(dtype),swiglu_limit_(swiglu_limit),hidden_(hidden),local_intermediate_(0){
  rt.assertOwnerThread();
  if(rt.isGraphRecording()||rt.device().type()!=runtime::Device::Type::kHygon||rank.device!=rt.device()||rank.tp_size<=0||rank.tp_rank<0||rank.tp_rank>=rank.tp_size
     ||!std::isfinite(swiglu_limit_)||swiglu_limit_<=0||!hidden||!intermediate||intermediate%size_t(rank.tp_size)||hidden%4
     ||(dtype!=runtime::DataType::kBFloat16&&dtype!=runtime::DataType::kFloat32))
    throw std::invalid_argument("DeepSeek INT8 MLP construction resources or geometry unsupported");
  local_intermediate_=intermediate/size_t(rank.tp_size);
  if(local_intermediate_%4||local_intermediate_>size_t(std::numeric_limits<int>::max()/2)||hidden>size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("DeepSeek INT8 MLP partition exceeds provider contract");
  try{
    w13_=runtime::Tensor::empty_parameter(rt,{2*local_intermediate_,hidden},runtime::DataType::kInt8);
    s13_=runtime::Tensor::empty_parameter(rt,{2*local_intermediate_,1},runtime::DataType::kFloat32);
    w2_=runtime::Tensor::empty_parameter(rt,{hidden,local_intermediate_},runtime::DataType::kInt8);
    s2_=runtime::Tensor::empty_parameter(rt,{hidden,1},runtime::DataType::kFloat32);
    for(size_t i=0;i<2;++i){
      const std::string name=i==0?"w1":"w3";
      bind_weight(name+".weight",weights::Weight(w13_->narrow({{0,i*local_intermediate_,local_intermediate_}}),0,rank.tp_rank,rank.tp_size));
      bind_weight(name+".scale",weights::Weight(s13_->narrow({{0,i*local_intermediate_,local_intermediate_}}),0,rank.tp_rank,rank.tp_size));
    }
    bind_weight("w2.weight",weights::Weight(w2_,1,rank.tp_rank,rank.tp_size));
    bind_weight("w2.scale",weights::Weight(s2_));
  }catch(...){try{rt.syncStream();}catch(...){std::terminate();}throw;}
}
DeepseekV4Int8MLPBankR20::DeepseekV4Int8MLPBankR20(runtime::Runtime&r,size_t hidden,size_t intermediate,
 const runtime::DataType&dtype,const engine::distributed::RankInfo&rank,
 runtime::Tensor w13,runtime::Tensor s13,runtime::Tensor w2,runtime::Tensor s2)
 :device_(r.device()),dtype_(dtype),swiglu_limit_(10.0),hidden_(hidden),local_intermediate_(256),
  w13_(std::move(w13)),s13_(std::move(s13)),w2_(std::move(w2)),s2_(std::move(s2)){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=runtime::Device::Type::kHygon||rank.device!=device_||rank.tp_size!=8||rank.tp_rank<0||rank.tp_rank>=8||hidden!=4096||intermediate!=2048||dtype!=runtime::DataType::kBFloat16)throw std::invalid_argument("DeepSeek bank MLP requires actual TP8 geometry");
 auto valid=[&](const runtime::Tensor&t,runtime::Shape shape,runtime::DataType type){return t&&t->device()==device_&&t->shape()==shape&&t->dtype()==type&&t->is_contiguous();};
 if(!valid(w13_,{512,4096},runtime::DataType::kInt8)||!valid(s13_,{512,1},runtime::DataType::kFloat32)||!valid(w2_,{4096,256},runtime::DataType::kInt8)||!valid(s2_,{4096,1},runtime::DataType::kFloat32))throw std::invalid_argument("DeepSeek bank MLP tensor geometry differs");
 for(size_t i=0;i<2;++i){const std::string name=i==0?"w1":"w3";
  bind_weight(name+".weight",weights::Weight(w13_->narrow({{0,i*256,256}}),0,rank.tp_rank,rank.tp_size));
  bind_weight(name+".scale",weights::Weight(s13_->narrow({{0,i*256,256}}),0,rank.tp_rank,rank.tp_size));
 }
 bind_weight("w2.weight",weights::Weight(w2_,1,rank.tp_rank,rank.tp_size));
 bind_weight("w2.scale",weights::Weight(s2_));
}
void DeepseekV4Int8MLPBankR20::process_weights_after_loading(runtime::Runtime &rt,bool){
  rt.assertOwnerThread();if(rt.device()!=device_||rt.isGraphRecording()||postload_attempted_)throw std::logic_error("DeepSeek INT8 MLP postload owner/state invalid");
  postload_attempted_=true;
  // Weight::load performs strict checkpoint dtype/shape validation. Parameters
  // remain in their original I8 and F32 scale layouts on this backend.
  rt.syncStream();ready_=true;
}
runtime::Tensor DeepseekV4Int8MLPBankR20::forward_local(runtime::Runtime &rt,const runtime::Tensor &input) const{
  rt.assertOwnerThread();
  if(!ready_||rt.device()!=device_||!input||input->device()!=device_||input->dtype()!=dtype_||input->ndim()!=2||input->size(1)!=hidden_)
    throw std::invalid_argument("DeepSeek INT8 MLP requires loaded same-owner [tokens,hidden] input");
  const size_t tokens=input->size(0);if(!tokens)throw std::invalid_argument("DeepSeek INT8 MLP zero token invocation unsupported");
  std::vector<runtime::Tensor> owners{input,w13_,s13_,w2_,s2_};
  try {
  auto linear=[&](const runtime::Tensor &x,const runtime::Tensor &w,const runtime::Tensor &scale){
    auto packed=runtime::Tensor::empty(rt,x->shape(),runtime::DataType::kInt8);
    owners.push_back(packed);
    auto xs=runtime::Tensor::empty(rt,{tokens,1},runtime::DataType::kFloat32);
    owners.push_back(xs);
    auto out=runtime::Tensor::empty(rt,{tokens,w->size(0)},dtype_);
    owners.push_back(out);
    auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(device_.type());quant.set_implementation_index(0);
    auto gemm=infini::ops::I8Gemm::DefaultConfig(device_.type());gemm.set_implementation_index(0);
    execution::submit<infini::ops::PerChannelQuantI8>(rt,execution::RecordingPolicy::HostOnly,quant,x,packed,xs);
    execution::submit<infini::ops::I8Gemm>(rt,execution::RecordingPolicy::HostOnly,gemm,out,packed,xs,w->permute({1,0}),scale,std::optional<runtime::Tensor>{});
    return out;
  };
  auto gate_up=linear(input,w13_,s13_);
  auto clamped=runtime::Tensor::empty(rt,gate_up->shape(),dtype_);
  owners.push_back(clamped);
  auto gate=gate_up->narrow({{1,0,local_intermediate_}}),up=gate_up->narrow({{1,local_intermediate_,local_intermediate_}});
  auto cg=clamped->narrow({{1,0,local_intermediate_}}),cu=clamped->narrow({{1,local_intermediate_,local_intermediate_}});
  execution::submit<infini::ops::Clamp>(rt,infini::ops::Clamp::DefaultConfig(device_.type()),gate,std::optional<double>{},std::optional<double>{swiglu_limit_},cg);
  execution::submit<infini::ops::Clamp>(rt,infini::ops::Clamp::DefaultConfig(device_.type()),up,std::optional<double>{-swiglu_limit_},std::optional<double>{swiglu_limit_},cu);
  auto activated=runtime::Tensor::empty(rt,{tokens,local_intermediate_},dtype_);
  owners.push_back(activated);
  execution::submit<infini::ops::SiluAndMul>(rt,infini::ops::SiluAndMul::DefaultConfig(device_.type()),clamped,activated);
  auto out=linear(activated,w2_,s2_);
  // One correctness fence retains every submitted operand through completion.
  // Replace with a worker-owned lease only after the asynchronous path is tested.
  rt.syncStream();return out;
  }catch(...){try{rt.syncStream();}catch(...){std::terminate();}throw;}
}
}
