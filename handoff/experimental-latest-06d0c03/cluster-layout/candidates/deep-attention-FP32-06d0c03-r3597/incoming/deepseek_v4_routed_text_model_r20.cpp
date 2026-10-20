#include "deep_reduce_wide_r3443.hpp"
#include "deepseek_v4_routed_text_model_r20.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <optional>
#include <set>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {
struct Work{
 rt::Runtime&r;std::vector<rt::Tensor>held;
 ~Work(){try{r.syncStream();}catch(...){std::terminate();}}
 rt::Tensor alloc(rt::Shape shape,rt::DataType type=rt::DataType::kFloat32){auto x=rt::Tensor::empty(r,shape,type);held.push_back(x);return x;}
 rt::Tensor ids(const std::vector<int64_t>&host){auto x=alloc({host.size()},rt::DataType::kInt64);r.memcpyH2D(x->data(),host.data(),host.size()*8,false);return x;}
 rt::Tensor cast(const rt::Tensor&x,rt::DataType type){if(x->dtype()==type)return x;auto y=alloc(x->shape(),type);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),x,y);return y;}
 template<class Op>rt::Tensor binary(const rt::Tensor&a,const rt::Tensor&b,rt::Shape shape){auto y=alloc(shape);ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,b,y);return y;}
 template<class Op>rt::Tensor unary(const rt::Tensor&a){auto y=alloc(a->shape());ex::submit<Op>(r,Op::DefaultConfig(r.device().type()),a,y);return y;}
};
void owner(rt::Runtime&r,rt::Runtime*expected){r.assertOwnerThread();if(&r!=expected||r.isGraphRecording())throw std::invalid_argument("DeepSeek text stack Runtime owner/recording differs");}
}
TextHeadR20::TextHeadR20(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):owner_(&r),rank_(rank),vocab_(cfg.at("vocab_size")){
 owner(r,owner_);
 if(r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||(rank.tp_size!=1&&rank.tp_size!=8)||rank.tp_rank<0||rank.tp_rank>=rank.tp_size||(rank.tp_size>1&&!rank.comm)||cfg.at("hidden_size")!=4096||cfg.at("hc_mult")!=4||vocab_!=129280||vocab_%rank.tp_size)throw std::invalid_argument("DeepSeek actual mHC text head geometry differs");
 fn_=rt::Tensor::empty_parameter(r,{4,16384},rt::DataType::kFloat32);base_=rt::Tensor::empty_parameter(r,{4},rt::DataType::kFloat32);scale_=rt::Tensor::empty_parameter(r,{1},rt::DataType::kFloat32);
 bind_weight("hc_head_fn",weights::Weight(fn_));bind_weight("hc_head_base",weights::Weight(base_));bind_weight("hc_head_scale",weights::Weight(scale_));
 norm_=std::make_unique<layers::RMSNorm>(r,4096,cfg.at("rms_norm_eps"),rt::DataType::kBFloat16);for(auto&item:norm_->state_dict())bind_weight("norm."+item.first,item.second);
 head_=rt::Tensor::empty_parameter(r,{vocab_/size_t(rank.tp_size),4096},rt::DataType::kBFloat16);bind_weight("head.weight",weights::Weight(head_,0,rank.tp_rank,rank.tp_size));
 constants_=rt::Tensor::empty(r,{2},rt::DataType::kFloat32);float values[]={float(cfg.at("rms_norm_eps").get<double>()),float(cfg.at("hc_eps").get<double>())};r.memcpyH2D(constants_->data(),values,sizeof(values),false);r.syncStream();
}
void TextHeadR20::process_weights_after_loading(rt::Runtime&r,bool flag){owner(r,owner_);if(ready_)throw std::logic_error("DeepSeek head already loaded");norm_->process_weights_after_loading(r,flag);r.syncStream();ready_=true;}
TextHeadR20::Output TextHeadR20::forward(rt::Runtime&r,const rt::Tensor&residual)const{
 owner(r,owner_);
 if(!ready_||!residual||residual->device()!=r.device()||residual->dtype()!=rt::DataType::kBFloat16||residual->ndim()!=3||!residual->size(0)||residual->size(0)>4096||residual->shape()!=rt::Shape{residual->size(0),4,4096}||!residual->is_contiguous())throw std::invalid_argument("DeepSeek head needs loaded contiguous BF16 [T,4,4096]");
 Work work{r};size_t t=residual->size(0);auto xf=work.cast(residual,rt::DataType::kFloat32),flat=xf->view({t,16384});
 auto square=work.binary<infini::ops::Mul>(flat,flat,{t,16384}),mean=work.alloc({t,1});ex::submit<infini::ops::Mean>(r,infini::ops::Mean::DefaultConfig(r.device().type()),square,std::optional<std::vector<int64_t>>{{1}},true,std::optional<infini::ops::DataType>{},mean);
 auto inv=work.unary<infini::ops::Rsqrt>(work.binary<infini::ops::Add>(mean,constants_->narrow({{0,0,1}}),{t,1}));auto mixes=work.alloc({t,4});
 // The checkpoint reference computes raw dot products, then applies RMS.
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),flat,fn_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},mixes);
 auto scaled=work.binary<infini::ops::Mul>(work.binary<infini::ops::Mul>(mixes,inv,{t,4}),scale_,{t,4});auto logits=work.binary<infini::ops::Add>(scaled,base_,{t,4});auto pre=work.binary<infini::ops::Add>(work.unary<infini::ops::Sigmoid>(logits),constants_->narrow({{0,1,1}}),{t,4});
 auto weighted=work.binary<infini::ops::Mul>(xf,pre->view({t,4,1}),{t,4,4096}),summed=work.alloc({t,4096});ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),weighted,std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},summed);
 auto collapsed=work.cast(summed,rt::DataType::kBFloat16),normalized=norm_->forward(r,collapsed);work.held.push_back(normalized);size_t width=vocab_/size_t(rank_.tp_size);auto local=work.alloc({t,width},rt::DataType::kBFloat16);
 ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),normalized,head_->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},local);
 rt::Tensor output=local;
 if(rank_.tp_size>1){auto gathered=work.alloc({size_t(rank_.tp_size),t,width},rt::DataType::kBFloat16);engine::distributed::allgather_(r,gathered,local,rank_.comm);output=gathered->permute({1,0,2})->contiguous(r)->view({t,vocab_});work.held.push_back(output);}
 r.syncStream();return {collapsed,normalized,output,pre};
}
TextModelR20::TextModelR20(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank,size_t rows,size_t capacity):owner_(&r),rows_(rows),capacity_(capacity),vocab_(cfg.at("vocab_size")){
 owner(r,owner_);
 if(cfg.at("num_hidden_layers")!=43||cfg.at("hidden_size")!=4096||cfg.at("hc_mult")!=4||rank.pp_size!=1||rank.pp_stage!=0||rank.world_size!=rank.tp_size||rank.world_rank!=rank.tp_rank||!rows||rows>128||!capacity||capacity>size_t(cfg.at("max_position_embeddings")))throw std::invalid_argument("DeepSeek text stack requires actual43-layer main model and PP1 topology");
 try{
  head_=std::make_unique<TextHeadR20>(r,cfg,rank);for(auto&item:head_->state_dict())bind_weight(item.first,item.second);
  embedding_=rt::Tensor::empty_parameter(r,{vocab_,4096},rt::DataType::kBFloat16);bind_weight("embed.weight",weights::Weight(embedding_));
  for(size_t layer=0;layer<43;++layer){auto bounded=cfg;size_t ratio=cfg.at("compress_ratios").at(layer);if(ratio)bounded["native_compressed_capacity"]=(capacity+ratio-1)/ratio;auto module=std::make_unique<NativeDecoderR20>(r,bounded,layer,rank,rows);for(auto&item:module->state_dict())bind_weight("layers."+std::to_string(layer)+"."+item.first,item.second);layers_.push_back(std::move(module));}
  lengths_.assign(rows,0);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void TextModelR20::process_weights_after_loading(rt::Runtime&r,bool flag){owner(r,owner_);if(ready_)throw std::logic_error("DeepSeek text model already loaded");for(auto&layer:layers_)layer->process_weights_after_loading(r,flag);head_->process_weights_after_loading(r,flag);r.syncStream();ready_=true;}
TextModelR20::Output TextModelR20::forward(rt::Runtime&r,const rt::Tensor&ids,const rt::Tensor&positions,const std::vector<Request>&requests,bool all_positions){
 owner(r,owner_);
 if(!ready_||poisoned_||!ids||!positions||ids->device()!=r.device()||positions->device()!=r.device()||ids->dtype()!=rt::DataType::kInt64||positions->dtype()!=rt::DataType::kInt64||ids->ndim()!=1||!ids->size(0)||ids->size(0)>4096||positions->shape()!=ids->shape()||!ids->is_contiguous()||!positions->is_contiguous()||requests.empty()||requests.size()>rows_)throw std::invalid_argument("DeepSeek text model requires loaded packed IDs/positions and valid state");
 size_t tokens=ids->size(0),total=0;std::set<size_t>seen;std::vector<int64_t>hi(tokens),hp(tokens),expanded,last;r.memcpyD2H(hi.data(),ids->data(),tokens*8);r.memcpyD2H(hp.data(),positions->data(),tokens*8);
 for(auto id:hi){if(id<0||size_t(id)>=vocab_)throw std::invalid_argument("DeepSeek text token outside vocabulary");for(size_t c=0;c<4;++c)expanded.push_back(id);}
 for(auto q:requests){if(q.row>=rows_||!seen.insert(q.row).second||!q.tokens||q.past!=lengths_[q.row]||q.past>capacity_||q.tokens>capacity_-q.past||q.tokens>tokens-total)throw std::invalid_argument("DeepSeek text packed row/past/count differs");for(size_t i=0;i<q.tokens;++i)if(hp[total+i]!=int64_t(q.past+i))throw std::invalid_argument("DeepSeek text logical position differs");total+=q.tokens;last.push_back(int64_t(total-1));}
 if(total!=tokens)throw std::invalid_argument("DeepSeek text packed count differs");
 wide_r3443::Scope request_scope_r3443(r,requests.size()); Work work{r};try{
  auto hidden=work.alloc({tokens*4,4096},rt::DataType::kBFloat16);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),embedding_,work.ids(expanded),int64_t{0},hidden);hidden=hidden->view({tokens,4,4096});
  for(auto&layer:layers_)hidden=layer->forward(r,hidden,ids,positions,requests).hidden;
  auto chosen=hidden;if(!all_positions){chosen=work.alloc({requests.size(),4,4096},rt::DataType::kBFloat16);ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),hidden,work.ids(last),int64_t{0},chosen);}
  auto result=head_->forward(r,chosen);for(auto q:requests)lengths_[q.row]=q.past+q.tokens;r.syncStream();return {hidden,result};
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void TextModelR20::reset(rt::Runtime&r){owner(r,owner_);for(auto&layer:layers_)layer->reset(r);std::fill(lengths_.begin(),lengths_.end(),0);poisoned_=false;}
}
