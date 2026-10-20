#include "deepseek_v4_attention_prepare.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}
AttentionPrepare::AttentionPrepare(rt::Runtime&r,const nlohmann::json&cfg,size_t layer,const engine::distributed::RankInfo&rank):device_(r.device()),local_heads_(0),max_position_(cfg.at("max_position_embeddings")),eps_(cfg.at("rms_norm_eps")),theta_(cfg.at("rope_theta")),factor_(1),beta_fast_(32),beta_slow_(1),original_(0){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||(rank.tp_size!=1&&rank.tp_size!=8)||rank.tp_rank<0||rank.tp_rank>=rank.tp_size||cfg.at("hidden_size")!=4096||cfg.at("head_dim")!=512||cfg.at("num_attention_heads")!=64||cfg.at("num_key_value_heads")!=1||cfg.at("q_lora_rank")!=1024||cfg.at("qk_rope_head_dim")!=64||layer>=cfg.at("num_hidden_layers").get<size_t>()||!max_position_||!std::isfinite(eps_)||eps_<=0)throw std::invalid_argument("DeepSeek attention prepare checkpoint geometry/owner differs");
 local_heads_=64/size_t(rank.tp_size);bool compressed=cfg.at("compress_ratios").at(layer).get<int>()!=0;
 if(compressed){theta_=cfg.at("compress_rope_theta");auto scaling=cfg.at("rope_scaling");factor_=scaling.at("factor");beta_fast_=scaling.at("beta_fast");beta_slow_=scaling.at("beta_slow");original_=scaling.at("original_max_position_embeddings");}
 if(!std::isfinite(theta_)||theta_<=1||!std::isfinite(factor_)||factor_<=0||beta_fast_<=0||beta_slow_<=0)throw std::invalid_argument("DeepSeek attention rotary config differs");
 try{
 auto bind=[&](const char*name,size_t rows,size_t columns,int split){auto weight=rt::Tensor::empty_parameter(r,{rows,columns},rt::DataType::kInt8);auto scale=rt::Tensor::empty_parameter(r,{rows,1},rt::DataType::kFloat32);bind_weight(std::string(name)+".weight",weights::Weight(weight,split,rank.tp_rank,rank.tp_size));bind_weight(std::string(name)+".scale",weights::Weight(scale,split,rank.tp_rank,rank.tp_size));return std::make_pair(weight,scale);};
 auto a=bind("wq_a",1024,4096,-1);qa_=a.first;qas_=a.second;auto k=bind("wkv",512,4096,-1);kv_=k.first;kvs_=k.second;auto b=bind("wq_b",local_heads_*512,1024,0);qb_=b.first;qbs_=b.second;
 q_norm_=bind_child("q_norm",std::make_shared<layers::RMSNorm>(r,1024,eps_,rt::DataType::kBFloat16));kv_norm_=bind_child("kv_norm",std::make_shared<layers::RMSNorm>(r,512,eps_,rt::DataType::kBFloat16));ones_=rt::Tensor::empty(r,{512},rt::DataType::kBFloat16);
 std::vector<float>one(512,1);auto f=rt::Tensor::empty(r,{512},rt::DataType::kFloat32);f->copy_from(r,rt::Tensor::from_blob(one.data(),{512},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}));ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),f,ones_);r.syncStream();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void AttentionPrepare::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("DeepSeek attention prepare postload state differs");r.syncStream();ready_=true;}
void AttentionPrepare::rotate(rt::Runtime&r,rt::Tensor q,std::optional<rt::Tensor>k,const rt::Tensor&pos,bool inverse)const{
 r.assertOwnerThread();if(r.device()!=device_||r.isGraphRecording()||!pos||pos->device()!=device_||pos->dtype()!=rt::DataType::kInt64||!pos->is_contiguous()||pos->shape()!=rt::Shape{q->size(0)})throw std::invalid_argument("DeepSeek rotary position owner/shape differs");
 size_t tokens=q->size(0);std::vector<int64_t>positions(tokens),rows(tokens);r.memcpyD2H(positions.data(),pos->data(),tokens*8);for(auto p:positions)if(p<0||size_t(p)>=max_position_)throw std::invalid_argument("DeepSeek rotary position out of bounds");std::iota(rows.begin(),rows.end(),0);
 constexpr double pi2=6.283185307179586476925286766559;double low=0,high=0;
 if(original_&&factor_!=1){auto correction=[&](double rotations){return 64*std::log(double(original_)/(rotations*pi2))/(2*std::log(theta_));};low=std::clamp(std::floor(correction(beta_fast_)),0.,63.);high=std::clamp(std::ceil(correction(beta_slow_)),0.,63.);if(low==high)high+=.001;}
 std::vector<float>cache(tokens*64);
 for(size_t t=0;t<tokens;++t)for(size_t i=0;i<32;++i){double frequency=1/std::pow(theta_,double(2*i)/64);if(original_&&factor_!=1){double smooth=1-std::clamp((double(i)-low)/(high-low),0.,1.);frequency=(frequency/factor_)*(1-smooth)+frequency*smooth;}double angle=double(positions[t])*frequency;cache[t*64+i]=float(std::cos(angle));cache[t*64+32+i]=float(std::sin(angle));}
 Work work{r};auto table=work.alloc({tokens,64},rt::DataType::kFloat32),ids=work.alloc({tokens},rt::DataType::kInt64);r.memcpyH2D(table->data(),cache.data(),cache.size()*4,false);r.memcpyH2D(ids->data(),rows.data(),rows.size()*8,false);
 ex::submit<infini::ops::RotaryEmbedding>(r,ex::RecordingPolicy::HostOnly,infini::ops::RotaryEmbedding::DefaultConfig(device_.type()),ids,q,k,table,int64_t{512},false,int64_t{448},inverse);r.syncStream();
}
AttentionPrepare::Output AttentionPrepare::forward(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&pos)const{
 r.assertOwnerThread();if(!ready_||r.device()!=device_||r.isGraphRecording()||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=4096)throw std::invalid_argument("DeepSeek attention prepare requires loaded BF16 hidden");Work work{r};size_t tokens=x->size(0);
 auto linear=[&](const rt::Tensor&input,const rt::Tensor&weight,const rt::Tensor&scale){auto packed=work.alloc(input->shape(),rt::DataType::kInt8),xs=work.alloc({tokens,1},rt::DataType::kFloat32),output=work.alloc({tokens,weight->size(0)},rt::DataType::kBFloat16);auto quant=infini::ops::PerChannelQuantI8::DefaultConfig(device_.type());quant.set_implementation_index(0);auto gemm=infini::ops::I8Gemm::DefaultConfig(device_.type());gemm.set_implementation_index(0);ex::submit<infini::ops::PerChannelQuantI8>(r,ex::RecordingPolicy::HostOnly,quant,input,packed,xs);ex::submit<infini::ops::I8Gemm>(r,ex::RecordingPolicy::HostOnly,gemm,output,packed,xs,weight->permute({1,0}),scale,std::optional<rt::Tensor>{});return output;};
 auto ql=q_norm_->forward(r,linear(x,qa_,qas_));work.held.push_back(ql);auto raw_q=linear(ql,qb_,qbs_)->view({tokens,local_heads_,512}),query=work.alloc({tokens,local_heads_,512},rt::DataType::kBFloat16);ex::submit<infini::ops::RmsNorm>(r,infini::ops::RmsNorm::DefaultConfig(device_.type()),raw_q,ones_,float(eps_),query);auto key=kv_norm_->forward(r,linear(x,kv_,kvs_));work.held.push_back(key);rotate(r,query,key->view({tokens,1,512}),pos,false);r.syncStream();return {query,key,ql};
}
rt::Tensor AttentionPrepare::inverse_output_rope(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&pos)const{
 r.assertOwnerThread();if(!ready_||r.device()!=device_||r.isGraphRecording()||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=3||!x->size(0)||x->size(1)!=local_heads_||x->size(2)!=512)throw std::invalid_argument("DeepSeek inverse rotary output geometry differs");Work work{r};auto result=work.alloc(x->shape(),x->dtype());result->copy_from(r,x);rotate(r,result,std::nullopt,pos,true);r.syncStream();return result;
}
}
