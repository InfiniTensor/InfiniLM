#include "glm_moe_dsa_projected_cache_r22.hpp"
#include "glm_selection_map_r302.hpp"
#include "glm_moe_dsa_indexer_rotary.hpp"
#include "glm_moe_dsa_sparse_attention.hpp"
#include "execution/submit.hpp"
#include <set>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <atomic>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {std::array<std::atomic<uint64_t>,8>view_calls927{},copy_calls927{};}
extern "C" uint64_t modern_glm_cache_view_calls_r927(int device,int kind){if(device<0||device>=8||kind<0||kind>1)return 0;return kind?copy_calls927[size_t(device)].load():view_calls927[size_t(device)].load();}
rt::Tensor sparse_attention_all_selected_decode_r601(rt::Runtime&,const rt::Tensor&,const rt::Tensor&,const rt::Tensor&,float);
namespace {
struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}rt::Tensor ids(const std::vector<int64_t>&v){auto x=alloc({v.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),v.data(),v.size()*8,false);return x;}};
}
NativeAttentionProjectedR22::NativeAttentionProjectedR22(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank,size_t rows,size_t capacity,size_t topk):owner_(&r),rows_(rows),capacity_(capacity),maximum_(cfg.at("max_position_embeddings")),topk_(topk),full_(cfg.at("indexer_types").at(layer)=="full"){
 r.assertOwnerThread();if(r.isGraphRecording()||!rows||rows>128||!capacity||capacity>maximum_||!topk||topk>2048||(cfg.at("indexer_types").at(layer)!="full"&&cfg.at("indexer_types").at(layer)!="shared"))throw std::invalid_argument("GLM persistent attention configuration differs");
 try{prepare_=std::make_shared<AttentionPrepare>(r,cfg,rank);output_=std::make_shared<AttentionOutput>(r,cfg,rank);for(auto&item:prepare_->state_dict())bind_weight(item.first,item.second);for(auto&item:output_->state_dict())bind_weight(item.first,item.second);
  if(full_){indexer_=std::make_shared<NativeIndexer>(r,cfg,rank);for(auto&item:indexer_->state_dict())bind_weight("indexer."+item.first,item.second);selector_=std::make_unique<IndexerSelection>(r,rows,capacity,32,128,topk,rank);r.setDeviceMemory(selector_->cache()->data(),0xff,selector_->cache()->nbytes());}
  latent_=rt::Tensor::empty(r,{rows,capacity,512},rt::DataType::kBFloat16);rotary_=rt::Tensor::empty(r,{rows,capacity,64},rt::DataType::kBFloat16);r.setDeviceMemory(latent_->data(),0xff,latent_->nbytes());r.setDeviceMemory(rotary_->data(),0xff,rotary_->nbytes());// Keep the original compressed cache contract and add model-owned projected
  // BF16 K/V banks. Only validated prefix lengths authorize reading a bank row.
  projected_key_=rt::Tensor::empty(r,{rows,capacity,prepare_->local_heads(),256},rt::DataType::kBFloat16);
  projected_value_=rt::Tensor::empty(r,{rows,capacity,prepare_->local_heads(),256},rt::DataType::kBFloat16);
  r.setDeviceMemory(projected_key_->data(),0xff,projected_key_->nbytes());
  r.setDeviceMemory(projected_value_->data(),0xff,projected_value_->nbytes());
  lengths_.assign(rows,0);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void NativeAttentionProjectedR22::process_weights_after_loading(rt::Runtime&r,bool flag){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::invalid_argument("GLM persistent attention postload differs");prepare_->process_weights_after_loading(r,flag);output_->process_weights_after_loading(r,flag);if(indexer_)indexer_->process_weights_after_loading(r,flag);ready_=true;}
void NativeAttentionProjectedR22::reset(rt::Runtime&r){r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::invalid_argument("GLM persistent attention reset owner differs");r.syncStream();if(selector_)selector_->reset(r);std::fill(lengths_.begin(),lengths_.end(),0);poisoned_=false;}
NativeAttentionProjectedR22::Output NativeAttentionProjectedR22::forward(rt::Runtime&r,const rt::Tensor&hidden,const rt::Tensor&positions,const std::vector<Request>&requests,const std::vector<rt::Tensor>&shared){
 r.assertOwnerThread();if(&r!=owner_||!ready_||poisoned_||r.isGraphRecording()||!hidden||hidden->device()!=r.device()||hidden->dtype()!=rt::DataType::kBFloat16||hidden->ndim()!=2||!hidden->is_contiguous()||!hidden->size(0)||hidden->size(0)>4096||hidden->size(1)!=6144||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||!positions->is_contiguous()||positions->shape()!=rt::Shape{hidden->size(0)}||requests.empty()||(full_?!shared.empty():shared.size()!=requests.size()))throw std::invalid_argument("GLM persistent attention input/selection differs");
 size_t tokens=hidden->size(0),total=0;std::set<size_t>seen;std::vector<int64_t>pos(tokens);r.memcpyD2H(pos.data(),positions->data(),tokens*8);
 // Reject all external metadata before either indexer or latent cache mutates.
 for(size_t i=0;i<requests.size();++i){auto request=requests[i];if(request.row>=rows_||!seen.insert(request.row).second||!request.tokens||request.past!=lengths_[request.row]||request.past>capacity_||request.tokens>capacity_-request.past||request.tokens>tokens-total)throw std::invalid_argument("GLM persistent attention request row/past/count differs");for(size_t t=0;t<request.tokens;++t)if(pos[total+t]!=int64_t(request.past+t))throw std::invalid_argument("GLM persistent attention positions differ");
  if(!full_){auto ids=shared[i];if(!ids||ids->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||!ids->is_contiguous()||ids->ndim()!=2||ids->size(0)!=request.tokens||!ids->size(1)||ids->size(1)>topk_)throw std::invalid_argument("GLM shared indexer selection shape differs");std::vector<int64_t>host(ids->numel());r.memcpyD2H(host.data(),ids->data(),host.size()*8);for(size_t t=0;t<request.tokens;++t){bool causal=false;for(size_t j=0;j<ids->size(1);++j){auto id=host[t*ids->size(1)+j];if(id<0||size_t(id)>=request.past+request.tokens)throw std::invalid_argument("GLM shared selection outside valid prefix");causal|=size_t(id)<=request.past+t;}if(!causal)throw std::invalid_argument("GLM shared selection has no causal entry");}}
  total+=request.tokens;
 }
 if(total!=tokens)throw std::invalid_argument("GLM persistent attention packed count differs");
 Work work{r};try{auto prepared=prepare_->forward_latent(r,hidden,positions);work.held.insert(work.held.end(),{prepared.query,prepared.q_resid,prepared.kv_latent,prepared.k_rot});std::vector<rt::Tensor>selection;
  if(full_){auto features=indexer_->forward(r,hidden,prepared.q_resid,positions);work.held.insert(work.held.end(),{features.query,features.key,features.head_weights});auto results=selector_->forward(r,features.query,features.key,features.head_weights,requests);for(auto&result:results){work.held.insert(work.held.end(),{result.scores,result.indices});selection.push_back(result.indices);}}
  else selection=shared;
  // Project independent arriving KV rows once for the complete packed batch.
  // Request validation and selection stay above; per-request cache writes,
  // sparse selection/order and length publication remain unchanged.
  auto fresh_all=prepare_->expand_cached(r,prepared.kv_latent,prepared.k_rot);
  work.held.insert(work.held.end(),{fresh_all.first,fresh_all.second});
  auto attention=work.alloc({tokens,prepare_->local_heads(),256},rt::DataType::kBFloat16);size_t begin=0;
  for(size_t i=0;i<requests.size();++i){auto request=requests[i];size_t count=request.tokens,prefix=request.past+count;auto latent_row=latent_->narrow({{0,request.row,1}})->view({capacity_,512}),rotary_row=rotary_->narrow({{0,request.row,1}})->view({capacity_,64});latent_row->narrow({{0,request.past,count}})->copy_from(r,prepared.kv_latent->narrow({{0,begin,count}}));rotary_row->narrow({{0,request.past,count}})->copy_from(r,prepared.k_rot->narrow({{0,begin,count}})->view({count,64}));
   auto key_row=projected_key_->narrow({{0,request.row,1}})->view({capacity_,prepare_->local_heads(),256});
   auto value_row=projected_value_->narrow({{0,request.row,1}})->view({capacity_,prepare_->local_heads(),256});
   // Project each arriving latent exactly once, retaining the same FP8
   // activation quantization, BF16 outputs, and rotary assembly as the baseline.
   std::pair<rt::Tensor,rt::Tensor>fresh{fresh_all.first->narrow({{0,begin,count}}),fresh_all.second->narrow({{0,begin,count}})};
   key_row->narrow({{0,request.past,count}})->copy_from(r,fresh.first);
   value_row->narrow({{0,request.past,count}})->copy_from(r,fresh.second);
   auto chosen=selection[i];std::vector<int64_t>host(chosen->numel());r.memcpyD2H(host.data(),chosen->data(),host.size()*8);auto remapped=selection_map_r302(host,count,chosen->size(1),request.past);auto&logical=remapped.logical;auto&mapped=remapped.mapped;
   auto ids=work.ids(logical);
   // Validated source-ordered selection_map output is the complete valid
   // prefix only if every logical position is present exactly once. Borrow
   // immutable contiguous cache views in that case; never expose spare slots.
   bool complete_prefix=logical.size()==prefix;
   for(size_t j=0;complete_prefix&&j<logical.size();++j)complete_prefix=logical[j]==int64_t(j);
   std::pair<rt::Tensor,rt::Tensor> expanded;
   if(complete_prefix){
    expanded={key_row->narrow({{0,0,prefix}}),value_row->narrow({{0,0,prefix}})};
    work.held.insert(work.held.end(),{expanded.first,expanded.second});
    view_calls927.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);
   }else{
    expanded={work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16),work.alloc({logical.size(),prepare_->local_heads(),256},rt::DataType::kBFloat16)};
    ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),key_row,ids,int64_t{0},expanded.first);
    ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),value_row,ids,int64_t{0},expanded.second);
    copy_calls927.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);
   }
   rt::Tensor result;
   if(count==1){result=sparse_attention_all_selected_decode_r601(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,1.f/16);}
   else{auto mapped_ids=work.ids(mapped)->view(chosen->shape());result=sparse_attention_mapped_bounded(r,prepared.query->narrow({{0,begin,count}}),expanded.first,expanded.second,mapped_ids,ids,request.past,1.f/16);}
   work.held.push_back(result);attention->narrow({{0,begin,count}})->copy_from(r,result);begin+=count;
  }
  auto projected=output_->forward(r,attention);work.held.insert(work.held.end(),{projected.local,projected.hidden});r.syncStream();for(auto request:requests)lengths_[request.row]=request.past+request.tokens;return {projected.hidden,selection,prepared,attention};
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
