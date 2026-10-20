#include "deepseek_v4_swa_attention.hpp"
#include "deepseek_v4_latent_attention.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <numeric>
#include <set>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}
SwaAttention::SwaAttention(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank,size_t rows):owner_(&r),rows_(rows),window_(cfg.at("sliding_window")),maximum_(cfg.at("max_position_embeddings")){
 r.assertOwnerThread();if(!rows||rows>128||window_!=128||layer>=2||cfg.at("compress_ratios").at(layer)!=0)throw std::invalid_argument("Native DeepSeek SWA qualifies noncompressed layers0/1 and window128");
 try{lengths_.assign(rows_,0);prepare_=std::make_shared<AttentionPrepare>(r,cfg,layer,rank);output_=std::make_shared<AttentionOutput>(r,cfg,rank);for(auto&item:prepare_->state_dict())bind_weight(item.first,item.second);for(auto&item:output_->state_dict())bind_weight(item.first,item.second);
 sink_=rt::Tensor::empty_parameter(r,{prepare_->local_heads()},rt::DataType::kFloat32);bind_weight("attn_sink",weights::Weight(sink_,0,rank.tp_rank,rank.tp_size));cache_=std::make_unique<PagedLatentCache>(r,(rows_*window_+255)/256,256);
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void SwaAttention::process_weights_after_loading(rt::Runtime&r,bool flag){r.assertOwnerThread();if(&r!=owner_||ready_||r.isGraphRecording())throw std::logic_error("DeepSeek SWA postload owner/state differs");prepare_->process_weights_after_loading(r,flag);output_->process_weights_after_loading(r,flag);ready_=true;}
void SwaAttention::reset(rt::Runtime&r){r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::logic_error("DeepSeek SWA reset owner differs");cache_->reset(r);std::fill(lengths_.begin(),lengths_.end(),0);}
rt::Tensor SwaAttention::forward(rt::Runtime&r,const rt::Tensor&hidden,const rt::Tensor&positions,const std::vector<Request>&requests){
 r.assertOwnerThread();if(&r!=owner_||!ready_||r.isGraphRecording()||!hidden||hidden->device()!=r.device()||hidden->dtype()!=rt::DataType::kBFloat16||hidden->ndim()!=2||!hidden->size(0)||hidden->size(0)>4096||hidden->size(1)!=4096||!positions||positions->device()!=r.device()||positions->dtype()!=rt::DataType::kInt64||positions->shape()!=rt::Shape{hidden->size(0)}||!positions->is_contiguous()||requests.empty())throw std::invalid_argument("DeepSeek SWA input/metadata contract differs");
 size_t tokens=hidden->size(0),total=0,old_total=0;std::set<size_t>seen;std::vector<int64_t>host_positions(tokens);r.memcpyD2H(host_positions.data(),positions->data(),tokens*8);
 for(auto request:requests){if(request.row>=rows_||!seen.insert(request.row).second||!request.tokens||request.past!=lengths_[request.row]||request.past>maximum_||request.tokens>maximum_-request.past||request.tokens>tokens-total)throw std::invalid_argument("DeepSeek SWA row/past/token metadata differs");for(size_t t=0;t<request.tokens;++t)if(host_positions[total+t]!=int64_t(request.past+t))throw std::invalid_argument("DeepSeek SWA positions must match each request's logical prefix");total+=request.tokens;old_total+=std::min(request.past,window_-1);}
 if(total!=tokens)throw std::invalid_argument("DeepSeek SWA packed token count differs");
 Work work{r};auto prep=prepare_->forward(r,hidden,positions);work.held.insert(work.held.end(),{prep.query,prep.kv,prep.q_lora});
 auto device_ids=[&](const std::vector<int64_t>&host){auto ids=work.alloc({host.size()},rt::DataType::kInt64);r.memcpyH2D(ids->data(),host.data(),host.size()*8,false);return ids;};
 std::vector<int64_t>new_slots(tokens);std::iota(new_slots.begin(),new_slots.end(),0);PagedLatentCache staging(r,(tokens+255)/256,256);auto staged_ids=device_ids(new_slots);staging.store(r,staged_ids,prep.kv);auto decoded=staging.gather(r,staged_ids);work.held.push_back(decoded);
 std::vector<int64_t>old_slots;old_slots.reserve(old_total);for(auto request:requests){size_t n=std::min(request.past,window_-1);for(size_t p=request.past-n;p<request.past;++p)old_slots.push_back(int64_t(request.row*window_+p%window_));}
 rt::Tensor old;if(old_total){old=cache_->gather(r,device_ids(old_slots));work.held.push_back(old);}
 auto bank=work.alloc({old_total+tokens,512},rt::DataType::kBFloat16);std::vector<int64_t>selected(tokens*window_,-1),ci(tokens,-1),commit_rows,commit_slots;std::vector<int32_t>lengths(tokens*2,0);size_t query_begin=0,bank_begin=0,old_begin=0;
 for(auto request:requests){size_t old_count=std::min(request.past,window_-1);if(old_count)bank->narrow({{0,bank_begin,old_count}})->copy_from(r,old->narrow({{0,old_begin,old_count}}));bank->narrow({{0,bank_begin+old_count,request.tokens}})->copy_from(r,decoded->narrow({{0,query_begin,request.tokens}}));
  for(size_t t=0;t<request.tokens;++t){size_t end=old_count+t+1,begin=end>window_?end-window_:0;lengths[(query_begin+t)*2]=int32_t(end-begin);for(size_t j=begin;j<end;++j)selected[(query_begin+t)*window_+j-begin]=int64_t(bank_begin+j);}
  size_t first=request.tokens>window_?request.tokens-window_:0;for(size_t t=first;t<request.tokens;++t){commit_rows.push_back(int64_t(query_begin+t));commit_slots.push_back(int64_t(request.row*window_+(request.past+t)%window_));}
  query_begin+=request.tokens;bank_begin+=old_count+request.tokens;old_begin+=old_count;
 }
 auto si=device_ids(selected)->view({tokens,window_}),compressed_indices=device_ids(ci)->view({tokens,1}),lens=work.alloc({tokens,2},rt::DataType::kInt32),unused=work.alloc({1,512},rt::DataType::kBFloat16);r.memcpyH2D(lens->data(),lengths.data(),lengths.size()*4,false);r.setDeviceMemory(unused->data(),0xff,unused->nbytes());
 auto attention=latent_attention(r,prep.query,bank,unused,sink_,si,compressed_indices,lens);work.held.push_back(attention);auto inverse=prepare_->inverse_output_rope(r,attention,positions);work.held.push_back(inverse);auto projected=output_->forward(r,inverse);work.held.insert(work.held.end(),{projected.grouped,projected.local,projected.hidden});
 // Commit only the last window from each request after all queries consumed
 // the old window. A long prefill must not overwrite keys needed by its start.
 auto commit=work.alloc({commit_rows.size(),512},rt::DataType::kBFloat16);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),prep.kv,device_ids(commit_rows),int64_t{0},commit);cache_->store(r,device_ids(commit_slots),commit);for(auto request:requests)lengths_[request.row]=request.past+request.tokens;r.syncStream();return projected.hidden;
}
}
