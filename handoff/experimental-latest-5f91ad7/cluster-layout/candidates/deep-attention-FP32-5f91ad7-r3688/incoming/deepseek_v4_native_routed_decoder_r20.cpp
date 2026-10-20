#include "deepseek_v4_native_routed_decoder_r20.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;
namespace {
void finalize(rt::Runtime&r,weights::ModelWeights&module,bool transpose){
 for(auto&child:module.children())finalize(r,*child.second,transpose);
 module.process_weights_after_loading(r,transpose);
}
template<class T>std::vector<typename T::Request>metadata(const std::vector<NativeDecoderR20::Request>&requests){
 std::vector<typename T::Request>result;for(auto q:requests)result.push_back({q.row,q.past,q.tokens});return result;
}
}
void NativeDecoderR20::check_owner(rt::Runtime&r)const{
 r.assertOwnerThread();if(owner_!=&r||r.isGraphRecording())throw std::invalid_argument("DeepSeek decoder Runtime owner differs");
}
NativeDecoderR20::NativeDecoderR20(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,
 const engine::distributed::RankInfo&rank,size_t rows):owner_(&r),rows_(rows),maximum_(cfg.at("max_position_embeddings")),vocab_(cfg.at("vocab_size")),ratio_(cfg.at("compress_ratios").at(layer)){
 check_owner(r);
 if(!rows||rows>128||layer>=size_t(cfg.at("num_hidden_layers"))||cfg.at("hidden_size")!=4096||cfg.at("hc_mult")!=4||cfg.at("n_routed_experts")!=256||cfg.at("num_experts_per_tok")!=6||cfg.at("n_shared_experts")!=1||cfg.at("swiglu_limit")!=10||cfg.at("scoring_func")!="sqrtsoftplus"||!cfg.at("norm_topk_prob").get<bool>()||(ratio_!=0&&ratio_!=4&&ratio_!=128))throw std::invalid_argument("DeepSeek native decoder checkpoint geometry differs");
 try{
  auto bind=[&](const std::string&prefix,weights::ModelWeights&module){for(auto&item:module.state_dict())bind_weight(prefix+item.first,item.second);};
  hc_=std::make_unique<HyperConnection>(r,cfg.at("rms_norm_eps"),cfg.at("hc_eps"),cfg.at("hc_sinkhorn_iters"));bind("hc_attn_",*hc_);
  norm_=std::make_unique<layers::RMSNorm>(r,4096,cfg.at("rms_norm_eps"),rt::DataType::kBFloat16);bind("attn_norm.",*norm_);
  if(ratio_==0){swa_=std::make_unique<SwaAttention>(r,cfg,layer,rank,rows);bind("attn.",*swa_);}
  else if(ratio_==4){csa_=std::make_unique<CsaAttention>(r,cfg,layer,rank,rows);bind("attn.",*csa_);}
  else{hca_=std::make_unique<HcaAttention>(r,cfg,layer,rank,rows);bind("attn.",*hca_);}
  size_t intermediate=cfg.at("moe_intermediate_size");bool hash=layer<size_t(cfg.at("num_hash_layers"));
  ffn_=std::make_unique<DeepseekV4FFNBlockR20>(r,intermediate,intermediate,vocab_,hash,cfg.at("routed_scaling_factor"),rank,cfg.at("rms_norm_eps"),cfg.at("hc_eps"),cfg.at("hc_sinkhorn_iters"));bind("",*ffn_);
  lengths_.assign(rows,0);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void NativeDecoderR20::process_weights_after_loading(rt::Runtime&r,bool flag){
 check_owner(r);if(ready_)throw std::logic_error("DeepSeek decoder already loaded");
 hc_->process_weights_after_loading(r,flag);norm_->process_weights_after_loading(r,flag);
 if(swa_)swa_->process_weights_after_loading(r,flag);if(csa_)csa_->process_weights_after_loading(r,flag);if(hca_)hca_->process_weights_after_loading(r,flag);
 finalize(r,*ffn_,flag);r.syncStream();ready_=true;
}
NativeDecoderR20::Output NativeDecoderR20::forward(rt::Runtime&r,const rt::Tensor&residual,const rt::Tensor&ids,const rt::Tensor&positions,const std::vector<Request>&requests){
 check_owner(r);
 if(!ready_||poisoned_||!residual||residual->device()!=r.device()||residual->dtype()!=rt::DataType::kBFloat16||residual->ndim()!=3||!residual->size(0)||residual->size(0)>4096||residual->shape()!=rt::Shape{residual->size(0),4,4096}||!residual->is_contiguous()||!ids||ids->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||ids->shape()!=rt::Shape{residual->size(0)}||!ids->is_contiguous()||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||positions->shape()!=ids->shape()||!positions->is_contiguous()||requests.empty()||requests.size()>rows_)throw std::invalid_argument("DeepSeek decoder needs loaded BF16 residual, packed IDs/positions and valid state");
 size_t tokens=residual->size(0),total=0;std::set<size_t>seen;std::vector<int64_t>host_ids(tokens),host_pos(tokens);
 r.memcpyD2H(host_ids.data(),ids->data(),tokens*8);r.memcpyD2H(host_pos.data(),positions->data(),tokens*8);
 for(auto id:host_ids)if(id<0||size_t(id)>=vocab_)throw std::invalid_argument("DeepSeek decoder token ID outside vocabulary");
 for(auto q:requests){if(q.row>=rows_||!seen.insert(q.row).second||!q.tokens||q.past!=lengths_[q.row]||q.past>maximum_||q.tokens>maximum_-q.past||q.tokens>tokens-total)throw std::invalid_argument("DeepSeek decoder packed row/past/count differs");for(size_t j=0;j<q.tokens;++j)if(host_pos[total+j]!=int64_t(q.past+j))throw std::invalid_argument("DeepSeek decoder logical position differs");total+=q.tokens;}
 if(total!=tokens)throw std::invalid_argument("DeepSeek decoder packed token count differs");
 try{
  Output out;out.attention_prepared=hc_->pre(r,residual);out.normalized_attention=norm_->forward(r,out.attention_prepared.collapsed);
  if(swa_)out.attention_value=swa_->forward(r,out.normalized_attention,positions,metadata<SwaAttention>(requests));
  else if(csa_)out.attention_value=csa_->forward(r,out.normalized_attention,positions,metadata<CsaAttention>(requests));
  else out.attention_value=hca_->forward(r,out.normalized_attention,positions,metadata<HcaAttention>(requests));
  out.attention_residual=hc_->post(r,out.attention_value,residual,out.attention_prepared);
  out.ffn=ffn_->forward_detailed(r,out.attention_residual,ids);out.hidden=out.ffn.value;
  for(auto q:requests)lengths_[q.row]=q.past+q.tokens;r.syncStream();return out;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void NativeDecoderR20::reset(rt::Runtime&r){check_owner(r);if(swa_)swa_->reset(r);if(csa_)csa_->reset(r);if(hca_)hca_->reset(r);std::fill(lengths_.begin(),lengths_.end(),0);poisoned_=false;}
std::vector<rt::Tensor>NativeDecoderR20::cache_tensors()const{
 if(swa_)return {swa_->raw_cache()};
 if(csa_)return {csa_->raw_cache(),csa_->raw_compressed_cache(),csa_->compressor_state(),csa_->raw_index_cache(),csa_->indexer_state()};
 return {hca_->raw_cache(),hca_->raw_compressed_cache(),hca_->compressor_state()};
}
}
