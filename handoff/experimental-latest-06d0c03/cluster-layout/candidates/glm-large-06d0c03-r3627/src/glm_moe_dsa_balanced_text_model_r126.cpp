#include "glm_moe_dsa_projected_text_model_r22.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <chrono>
#include <cuda_runtime_api.h>
namespace {thread_local double glm_r16_profile[6]={};using GlmClock=std::chrono::steady_clock;}
extern "C" const double*modern_glm_pipeline_last_profile(){return glm_r16_profile;}
namespace {
struct GlmGpuWindows {
 cudaEvent_t event[4]{};cudaStream_t stream;
 explicit GlmGpuWindows(cudaStream_t s):stream(s){try{for(auto&e:event)if(cudaEventCreate(&e)!=cudaSuccess)throw std::runtime_error("GLM profile event creation failed");}catch(...){for(auto e:event)if(e)cudaEventDestroy(e);throw;}}
 ~GlmGpuWindows(){for(auto e:event)if(e)cudaEventDestroy(e);}
 void record(int i){if(cudaEventRecord(event[i],stream)!=cudaSuccess)throw std::runtime_error("GLM profile event record failed");}
 void finish(){for(int i=0;i<3;++i){float ms=0;if(cudaEventElapsedTime(&ms,event[i],event[i+1])!=cudaSuccess)throw std::runtime_error("GLM profile event elapsed failed");glm_r16_profile[3+i]=double(ms)/1000.;}}
};
}


#include <numeric>
#include <set>
#include <stdexcept>

namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
struct Work {
 rt::Runtime&r;std::vector<rt::Tensor>held;
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor alloc(rt::Shape shape,rt::DataType type=rt::DataType::kBFloat16){auto x=rt::Tensor::empty(r,shape,type);held.push_back(x);return x;}
 rt::Tensor ids(const std::vector<int64_t>&host){auto x=alloc({host.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),host.data(),host.size()*8,false);return x;}
};
}
void TextModelProjectedR22::check_owner(rt::Runtime&r)const{r.assertOwnerThread();if(owner_!=&r||r.isGraphRecording())throw std::invalid_argument("GLM text stack Runtime owner differs");}
TextModelProjectedR22::TextModelProjectedR22(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank,size_t rows,size_t capacity,const bool*bounded_policy,const bool*owned_prefix_policy):owner_(&r),rank_(rank),cfg_(cfg),rows_(rows),capacity_(capacity),vocab_(cfg.at("vocab_size")),begin_(0),end_(0),eps_(cfg.at("rms_norm_eps")){
 check_owner(r);
 if(r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(rank.tp_size!=1&&rank.tp_size!=8)||rank.tp_rank<0||rank.tp_rank>=rank.tp_size||(rank.pp_size!=1&&rank.pp_size!=2&&rank.pp_size!=4)||rank.pp_stage<0||rank.pp_stage>=rank.pp_size||rank.world_size!=rank.tp_size*rank.pp_size||rank.world_rank!=rank.pp_stage*rank.tp_size+rank.tp_rank||(rank.tp_size>1&&!rank.comm)||(rank.pp_size>1&&!rank.world_comm)||cfg.at("num_hidden_layers")!=78||cfg.at("hidden_size")!=6144||vocab_!=154880||vocab_%rank.tp_size||!rows||rows>192||!capacity||capacity>size_t(cfg.at("max_position_embeddings")))throw std::invalid_argument("GLM full text stack requires actual78 layers and explicit TP1/8 PP1/2/4 topology");
 const size_t pp4[]={0,18,38,58,78},pp2[]={0,38,78};const size_t*boundaries=rank.pp_size==2?pp2:pp4;begin_=rank.pp_size==1?0:boundaries[rank.pp_stage];end_=rank.pp_size==1?78:boundaries[rank.pp_stage+1];
 if(cfg.at("indexer_types").at(begin_)!="full")throw std::invalid_argument("GLM PP stage must begin with a full indexer");
 try{
  if(first_stage()){embedding_=rt::Tensor::empty_parameter(r,{vocab_,6144},rt::DataType::kBFloat16);bind_weight("model.embed_tokens.weight",weights::Weight(embedding_));}
  for(size_t layer=begin_;layer<end_;++layer){auto module=std::make_unique<DecoderProjectedR22>(r,cfg,layer,rank,rows,capacity,2048,bounded_policy,owned_prefix_policy);for(auto&item:module->state_dict())bind_weight("model.layers."+std::to_string(layer)+"."+item.first,item.second);layers_.push_back(std::move(module));}
  if(last_stage()){
   norm_=rt::Tensor::empty_parameter(r,{6144},rt::DataType::kBFloat16);bind_weight("model.norm.weight",weights::Weight(norm_));
   head_=rt::Tensor::empty_parameter(r,{vocab_/size_t(rank.tp_size),6144},rt::DataType::kBFloat16);bind_weight("lm_head.weight",weights::Weight(head_,0,rank.tp_rank,rank.tp_size));
   ones_=rt::Tensor::empty(r,{6144},rt::DataType::kBFloat16);std::vector<uint16_t>ones(6144,0x3f80);r.memcpyH2D(ones_->data(),ones.data(),ones.size()*2,false);
  }
  lengths_.assign(rows,0);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void TextModelProjectedR22::process_weights_after_loading(rt::Runtime&r,bool flag){check_owner(r);if(ready_)throw std::logic_error("GLM text stack already loaded");for(auto&layer:layers_)layer->process_weights_after_loading(r,flag);r.syncStream();ready_=true;}
TextModelProjectedR22::Output TextModelProjectedR22::forward(rt::Runtime&r,const rt::Tensor&ids,const rt::Tensor&positions,const std::vector<NativeAttentionProjectedR22::Request>&requests,bool all_positions){
 check_owner(r);
 if(!ready_||poisoned_||!ids||!positions||ids->device()!=r.device()||positions->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||positions->dtype()!=rt::DataType::kInt64||ids->ndim()!=1||!ids->size(0)||ids->size(0)>4096||positions->shape()!=ids->shape()||!ids->is_contiguous()||!positions->is_contiguous()||requests.empty()||requests.size()>rows_)throw std::invalid_argument("GLM text stack requires loaded packed IDs/positions and valid state");
 size_t tokens=ids->size(0),total=0;std::set<size_t>seen;std::vector<int64_t>hi(tokens),hp(tokens),last;
 r.memcpyD2H(hi.data(),ids->data(),tokens*8);r.memcpyD2H(hp.data(),positions->data(),tokens*8);
 for(auto id:hi)if(id<0||size_t(id)>=vocab_)throw std::invalid_argument("GLM token ID outside embedding");
 for(auto q:requests){if(q.row>=rows_||!seen.insert(q.row).second||!q.tokens||q.past!=lengths_[q.row]||q.past>capacity_||q.tokens>capacity_-q.past||q.tokens>tokens-total)throw std::invalid_argument("GLM text stack row/past/count differs");for(size_t i=0;i<q.tokens;++i)if(hp[total+i]!=int64_t(q.past+i))throw std::invalid_argument("GLM text stack logical position differs");total+=q.tokens;last.push_back(int64_t(total-1));}
 if(total!=tokens)throw std::invalid_argument("GLM text stack packed count differs");
 Work work{r};try{
  GlmGpuWindows gpu_windows(static_cast<cudaStream_t>(r.stream()));gpu_windows.record(0);auto incoming_begin=GlmClock::now();rt::Tensor hidden;
  if(first_stage()){hidden=work.alloc({tokens,6144});ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),embedding_,ids,int64_t{0},hidden);}
  else{
   size_t shard=6144/size_t(rank_.tp_size);auto local=work.alloc({tokens,shard});engine::distributed::recv_(r,local,(rank_.pp_stage-1)*rank_.tp_size+rank_.tp_rank,rank_.world_comm);
   if(rank_.tp_size==1)hidden=local;
   else{auto gathered=work.alloc({size_t(rank_.tp_size),tokens,shard});engine::distributed::allgather_(r,gathered,local,rank_.comm);hidden=gathered->permute({1,0,2})->contiguous(r)->view({tokens,6144});work.held.push_back(hidden);}
  }
  gpu_windows.record(1);glm_r16_profile[0]=std::chrono::duration<double>(GlmClock::now()-incoming_begin).count();auto layers_begin=GlmClock::now();
  std::vector<rt::Tensor>selection;PrefixProofsR1645 selection_proofs;
  for(size_t i=0;i<layers_.size();++i){bool full=cfg_.at("indexer_types").at(begin_+i)=="full";if(full){selection.clear();selection_proofs.clear();}auto out=layers_[i]->forward(r,hidden,positions,requests,selection,selection_proofs);hidden=out.hidden;selection=std::move(out.attention.selection);selection_proofs=std::move(out.attention.selection_proofs);}
  gpu_windows.record(2);glm_r16_profile[1]=std::chrono::duration<double>(GlmClock::now()-layers_begin).count();auto outgoing_begin=GlmClock::now();
  rt::Tensor logits;
  if(!last_stage()){
   size_t shard=6144/size_t(rank_.tp_size);auto local=work.alloc({tokens,shard});local->copy_from(r,hidden->narrow({{1,size_t(rank_.tp_rank)*shard,shard}}));engine::distributed::send(r,local,(rank_.pp_stage+1)*rank_.tp_size+rank_.tp_rank,rank_.world_comm);
  }else{
   auto normalized=work.alloc(hidden->shape()),weighted=work.alloc(hidden->shape());ex::submit<infini::ops::RmsNorm>(r,infini::ops::RmsNorm::DefaultConfig(r.device().type()),hidden,ones_,eps_,normalized);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(r.device().type()),normalized,norm_,weighted);
   rt::Tensor chosen=weighted;
   if(!all_positions){chosen=work.alloc({requests.size(),6144});ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),weighted,work.ids(last),int64_t{0},chosen);}
   size_t n=chosen->size(0),width=vocab_/size_t(rank_.tp_size);auto local=work.alloc({n,width});ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),chosen,head_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},local);
   if(rank_.tp_size==1)logits=local;
   else{auto gathered=work.alloc({size_t(rank_.tp_size),n,width});engine::distributed::allgather_(r,gathered,local,rank_.comm);logits=gathered->permute({1,0,2})->contiguous(r)->view({n,vocab_});work.held.push_back(logits);}
   hidden=weighted;
  }
  for(auto q:requests)lengths_[q.row]=q.past+q.tokens;gpu_windows.record(3);r.syncStream();gpu_windows.finish();glm_r16_profile[2]=std::chrono::duration<double>(GlmClock::now()-outgoing_begin).count();return {hidden,logits};
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void TextModelProjectedR22::reset(rt::Runtime&r){check_owner(r);for(auto&layer:layers_)layer->reset(r);std::fill(lengths_.begin(),lengths_.end(),0);poisoned_=false;}
}
