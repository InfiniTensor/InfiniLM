#include "deep_cache_batch_r784.hpp"
#include "deepseek_v4_csa_attention.hpp"
#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}
CsaAttention::CsaAttention(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank,size_t rows):owner_(&r),rows_(rows),window_(cfg.at("sliding_window")),maximum_(cfg.at("max_position_embeddings")),compressed_capacity_(cfg.value("native_compressed_capacity",size_t(maximum_/4))){
 r.assertOwnerThread();if(!rows||rows>128||window_!=128||cfg.at("compress_ratios").at(layer)!=4||!compressed_capacity_||compressed_capacity_>maximum_/4)throw std::invalid_argument("Native DeepSeek CSA qualifies noncompressed layers0/1 and window128");
 try{lengths_.assign(rows_,0);prepare_=std::make_shared<AttentionPrepare>(r,cfg,layer,rank);output_=std::make_shared<AttentionOutput>(r,cfg,rank);for(auto&item:prepare_->state_dict())bind_weight(item.first,item.second);for(auto&item:output_->state_dict())bind_weight(item.first,item.second);
 sink_=rt::Tensor::empty_parameter(r,{prepare_->local_heads()},rt::DataType::kFloat32);bind_weight("attn_sink",weights::Weight(sink_,0,rank.tp_rank,rank.tp_size));cache_=std::make_unique<PagedLatentCache>(r,(rows_*window_+255)/256,256);
 compressor_=std::make_unique<TemporalCompressor>(r,cfg,layer,512,rows_,compressed_capacity_);for(auto&item:compressor_->state_dict())bind_weight("compressor."+item.first,item.second);
 topk_=cfg.value("native_c4_index_topk",size_t(512));indexer_=std::make_unique<C4Indexer>(r,cfg,layer,rank,rows_,compressed_capacity_,topk_);for(auto&item:indexer_->state_dict())bind_weight("indexer."+item.first,item.second);
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void CsaAttention::process_weights_after_loading(rt::Runtime&r,bool flag){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("DeepSeek CSA postload owner/state differs");prepare_->process_weights_after_loading(r,flag);output_->process_weights_after_loading(r,flag);compressor_->process_weights_after_loading(r,flag);indexer_->process_weights_after_loading(r,flag);ready_=true;}
void CsaAttention::reset(rt::Runtime&r){r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::logic_error("DeepSeek CSA reset owner differs");cache_->reset(r);compressor_->reset(r);indexer_->reset(r);last_selection_.clear();poisoned_=false;std::fill(lengths_.begin(),lengths_.end(),0);}
rt::Tensor CsaAttention::forward(rt::Runtime&r,const rt::Tensor&hidden,const rt::Tensor&positions,const std::vector<Request>&requests){
 r.assertOwnerThread();if(&r!=owner_||!ready_||poisoned_||r.isGraphRecording()||!hidden||hidden->device()!=r.device()||hidden->dtype()!=rt::DataType::kBFloat16||hidden->ndim()!=2||!hidden->size(0)||hidden->size(0)>4096||hidden->size(1)!=4096||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||positions->shape()!=rt::Shape{hidden->size(0)}||!positions->is_contiguous()||requests.empty())throw std::invalid_argument("DeepSeek CSA input/metadata contract differs");
 size_t tokens=hidden->size(0),total=0,old_total=0;std::set<size_t>seen;std::vector<int64_t>host_positions(tokens);r.memcpyD2H(host_positions.data(),positions->data(),tokens*8);
 for(auto request:requests){if(request.row>=rows_||!seen.insert(request.row).second||!request.tokens||request.past!=lengths_[request.row]||request.past>maximum_||request.tokens>maximum_-request.past||request.tokens>tokens-total||(request.past+request.tokens)/4>compressed_capacity_)throw std::invalid_argument("DeepSeek CSA row/past/token metadata differs");for(size_t t=0;t<request.tokens;++t)if(host_positions[total+t]!=int64_t(request.past+t))throw std::invalid_argument("DeepSeek CSA positions must match each request's logical prefix");total+=request.tokens;old_total+=std::min(request.past,window_-1);}
 if(total!=tokens)throw std::invalid_argument("DeepSeek CSA packed token count differs");
 try{Work work{r};auto prep=prepare_->forward(r,hidden,positions);work.held.insert(work.held.end(),{prep.query,prep.kv,prep.q_lora});
 std::vector<TemporalCompressor::Request>compress_requests;size_t compressed_total=0,compressed_width=1;
 for(auto request:requests){compress_requests.push_back({request.row,request.past,request.tokens});size_t count=(request.past+request.tokens)/4;compressed_total+=count;compressed_width=std::max(compressed_width,count);}
 auto pending=compressor_->prepare(r,hidden,positions,compress_requests);
 auto index_pending=indexer_->prepare(r,hidden,prep.q_lora,positions,compress_requests);
 compressed_width=topk_;

 auto device_ids=[&](const std::vector<int64_t>&host){auto ids=work.alloc({host.size()},rt::DataType::kInt64);r.memcpyH2D(ids->data(),host.data(),host.size()*8,false);return ids;};
 std::vector<int64_t>new_slots(tokens);std::iota(new_slots.begin(),new_slots.end(),0);PagedLatentCache staging(r,(tokens+255)/256,256);auto staged_ids=device_ids(new_slots);staging.store(r,staged_ids,prep.kv);auto decoded=staging.gather(r,staged_ids);work.held.push_back(decoded);
 std::vector<int64_t>old_slots;old_slots.reserve(old_total);for(auto request:requests){size_t n=std::min(request.past,window_-1);for(size_t p=request.past-n;p<request.past;++p)old_slots.push_back(int64_t(request.row*window_+p%window_));}
 rt::Tensor old;if(old_total){old=cache_->gather(r,device_ids(old_slots));work.held.push_back(old);}
 
 auto compressed_bank=work.alloc({std::max(size_t(1),compressed_total),512},rt::DataType::kBFloat16);
 r.setDeviceMemory(compressed_bank->data(),0xff,compressed_bank->nbytes());
 rt::Tensor new_compressed;
 if(!pending.boundaries.empty()){
  PagedLatentCache compressed_staging(r,(pending.boundaries.size()+63)/64,64);
  std::vector<int64_t>staged(pending.boundaries.size());std::iota(staged.begin(),staged.end(),0);
  auto ids=device_ids(staged);compressed_staging.store(r,ids,pending.rotated);new_compressed=compressed_staging.gather(r,ids);work.held.push_back(new_compressed);
 }
 size_t historical_count=0;for(auto request:requests)historical_count+=request.past/4;
 rt::Tensor historical_batch;
 if(requests.size()>1&&historical_count&&historical_count<=65536){
  std::vector<std::pair<size_t,size_t>>entries;entries.reserve(historical_count);
  for(auto request:requests)for(size_t i=0;i<request.past/4;++i)entries.push_back({request.row,i});
  historical_batch=compressor_->gather(r,entries);work.held.push_back(historical_batch);
  modern_deep_cache_batch_count_r784(1,r.device().index());
 }
 size_t historical_begin=0;
 size_t compressed_begin=0,new_compressed_begin=0;
 for(auto request:requests){size_t before=request.past/4,after=(request.past+request.tokens)/4;
  if(before){for(size_t start=0;start<before;start+=65536){size_t n=std::min(size_t(65536),before-start);std::vector<std::pair<size_t,size_t>>entries;for(size_t i=start;i<start+n;++i)entries.push_back({request.row,i});auto old_compressed=historical_batch?historical_batch->narrow({{0,historical_begin+start,n}}):compressor_->gather(r,entries);work.held.push_back(old_compressed);compressed_bank->narrow({{0,compressed_begin+start,n}})->copy_from(r,old_compressed);}}
  if(after>before)compressed_bank->narrow({{0,compressed_begin+before,after-before}})->copy_from(r,new_compressed->narrow({{0,new_compressed_begin,after-before}}));
  compressed_begin+=after;new_compressed_begin+=after-before;historical_begin+=before;
 }
auto bank=work.alloc({old_total+tokens,512},rt::DataType::kBFloat16);std::vector<int64_t>selected(tokens*window_,-1),ci(tokens*compressed_width,-1),commit_rows,commit_slots;std::vector<int32_t>lengths(tokens*2,0);size_t query_begin=0,bank_begin=0,old_begin=0;
 for(auto request:requests){size_t old_count=std::min(request.past,window_-1);if(old_count)bank->narrow({{0,bank_begin,old_count}})->copy_from(r,old->narrow({{0,old_begin,old_count}}));bank->narrow({{0,bank_begin+old_count,request.tokens}})->copy_from(r,decoded->narrow({{0,query_begin,request.tokens}}));
  for(size_t t=0;t<request.tokens;++t){size_t end=old_count+t+1,begin=end>window_?end-window_:0;lengths[(query_begin+t)*2]=int32_t(end-begin);for(size_t j=begin;j<end;++j)selected[(query_begin+t)*window_+j-begin]=int64_t(bank_begin+j);}
  size_t first=request.tokens>window_?request.tokens-window_:0;for(size_t t=first;t<request.tokens;++t){commit_rows.push_back(int64_t(query_begin+t));commit_slots.push_back(int64_t(request.row*window_+(request.past+t)%window_));}
  query_begin+=request.tokens;bank_begin+=old_count+request.tokens;old_begin+=old_count;
 }
 
 query_begin=0;compressed_begin=0;
 size_t request_index=0;
 for(auto request:requests){auto selected=index_pending.indices.at(request_index++);size_t width=selected->size(1);std::vector<int64_t>host(request.tokens*width);r.memcpyD2H(host.data(),selected->data(),host.size()*8);
  for(size_t t=0;t<request.tokens;++t){size_t visible=(request.past+t+1)/4,count=0;std::set<int64_t>seen;
   for(size_t j=0;j<width;++j){auto id=host[t*width+j];if(id<0||size_t(id)>=visible)continue;if(!seen.insert(id).second)throw std::invalid_argument("DeepSeek CSA duplicate C4 selection");ci[(query_begin+t)*compressed_width+count++]=int64_t(compressed_begin)+id;}
   if(count!=std::min(topk_,visible))throw std::invalid_argument("DeepSeek CSA incomplete causal C4 selection");lengths[(query_begin+t)*2+1]=int32_t(count);
  }query_begin+=request.tokens;compressed_begin+=(request.past+request.tokens)/4;
 }
 auto si=device_ids(selected)->view({tokens,window_}),compressed_indices=device_ids(ci)->view({tokens,compressed_width}),lens=work.alloc({tokens,2},rt::DataType::kInt32),unused=work.alloc({1,512},rt::DataType::kBFloat16);r.memcpyH2D(lens->data(),lengths.data(),lengths.size()*4,false);r.setDeviceMemory(unused->data(),0xff,unused->nbytes());
 auto attention=latent_attention(r,prep.query,bank,compressed_bank,sink_,si,compressed_indices,lens);work.held.push_back(attention);auto inverse=prepare_->inverse_output_rope(r,attention,positions);work.held.push_back(inverse);auto projected=output_->forward(r,inverse);work.held.insert(work.held.end(),{projected.grouped,projected.local,projected.hidden});
 // Commit only the last window from each request after all queries consumed
 // the old window. A long prefill must not overwrite keys needed by its start.
 auto commit=work.alloc({commit_rows.size(),512},rt::DataType::kBFloat16);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),prep.kv,device_ids(commit_rows),int64_t{0},commit);cache_->store(r,device_ids(commit_slots),commit);compressor_->commit(r,pending);indexer_->commit(r,index_pending);last_selection_=index_pending.indices;for(auto request:requests)lengths_[request.row]=request.past+request.tokens;r.syncStream();return projected.hidden;}catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
