#include "glm_moe_dsa_attention_prepare.hpp"
#include "glm_moe_dsa_indexer_rotary.hpp"
#include "execution/submit.hpp"
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa{
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace {struct Work{rt::Runtime&r;std::vector<rt::Tensor>held;~Work(){try{r.syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape s,rt::DataType d){auto x=rt::Tensor::empty(r,s,d);held.push_back(x);return x;}};}
Fp8Projection::Fp8Projection(rt::Runtime&r,size_t rows,size_t columns,int tp,int rank):device_(r.device()),rows_(0),columns_(columns),offset_(0){
 r.assertOwnerThread();if(!rows||!columns||tp<1||rank<0||rank>=tp||rows%size_t(tp)||r.isGraphRecording())throw std::invalid_argument("GLM FP8 projection partition differs");
 rows_=rows/size_t(tp);offset_=size_t(rank)*rows_;
 weight_=rt::Tensor::empty_parameter(r,{rows_,columns_},rt::DataType::kUInt8);scale_=rt::Tensor::empty_parameter(r,{(rows+127)/128,(columns+127)/128},rt::DataType::kFloat32);
 bind_weight("weight",weights::Weight(weight_,tp>1?0:-1,rank,tp));bind_weight("weight_scale_inv",weights::Weight(scale_,-1,rank,tp));
}
rt::Tensor Fp8Projection::forward(rt::Runtime&r,const rt::Tensor&x)const{
 r.assertOwnerThread();if(r.device()!=device_||r.isGraphRecording()||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=columns_||!x->is_contiguous())throw std::invalid_argument("GLM FP8 projection requires owned contiguous BF16 input");
 Work work{r};auto packed=work.alloc(x->shape(),rt::DataType::kUInt8),scale=work.alloc({x->size(0),(columns_+127)/128},rt::DataType::kFloat32),out=work.alloc({x->size(0),rows_},rt::DataType::kBFloat16);
 ex::submit<infini::ops::BlockQuantF8E4m3>(r,ex::RecordingPolicy::HostOnly,infini::ops::BlockQuantF8E4m3::DefaultConfig(device_.type()),x,packed,scale);
 ex::submit<infini::ops::Fp8BlockLinear>(r,ex::RecordingPolicy::HostOnly,infini::ops::Fp8BlockLinear::DefaultConfig(device_.type()),out,packed,scale,weight_,scale_,offset_,size_t{0});r.syncStream();return out;
}
AttentionPrepare::AttentionPrepare(rt::Runtime&r,const nlohmann::json&cfg,const engine::distributed::RankInfo&rank):device_(r.device()),heads_(0),maximum_(cfg.at("max_position_embeddings")){
 r.assertOwnerThread();if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||(rank.tp_size!=1&&rank.tp_size!=8)||rank.tp_rank<0||rank.tp_rank>=rank.tp_size||cfg.at("hidden_size")!=6144||cfg.at("num_attention_heads")!=64||cfg.at("q_lora_rank")!=2048||cfg.at("kv_lora_rank")!=512||cfg.at("qk_nope_head_dim")!=192||cfg.at("qk_rope_head_dim")!=64||cfg.at("v_head_dim")!=256||cfg.value("attention_bias",false))throw std::invalid_argument("GLM attention prepare checkpoint geometry/owner differs");
 heads_=64/size_t(rank.tp_size);
 try{
 qa_=bind_child("q_a_proj",std::make_shared<Fp8Projection>(r,2048,6144,1,0));qb_=bind_child("q_b_proj",std::make_shared<Fp8Projection>(r,16384,2048,rank.tp_size,rank.tp_rank));
 ka_=bind_child("kv_a_proj_with_mqa",std::make_shared<Fp8Projection>(r,576,6144,1,0));kb_=bind_child("kv_b_proj",std::make_shared<Fp8Projection>(r,28672,512,rank.tp_size,rank.tp_rank));
 qn_=rt::Tensor::empty_parameter(r,{2048},rt::DataType::kBFloat16);kn_=rt::Tensor::empty_parameter(r,{512},rt::DataType::kBFloat16);
 bind_weight("q_a_layernorm.weight",weights::Weight(qn_));bind_weight("kv_a_layernorm.weight",weights::Weight(kn_));
 auto ones=[&](size_t n){std::vector<float>host(n,1.f);auto f=rt::Tensor::empty(r,{n},rt::DataType::kFloat32),b=rt::Tensor::empty(r,{n},rt::DataType::kBFloat16);r.memcpyH2D(f->data(),host.data(),n*4,false);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),f,b);r.syncStream();return b;};qones_=ones(2048);kones_=ones(512);
 auto parameters=cfg.at("rope_parameters");if(parameters.value("rope_type",std::string("default"))!="default")throw std::invalid_argument("GLM non-default rotary scaling is not qualified");double theta=parameters.at("rope_theta");
 rope_=std::make_unique<layers::RoPE>(r,64,64,maximum_,theta,layers::RoPE::Algo::GPT_J,rt::DataType::kBFloat16,device_);
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void AttentionPrepare::process_weights_after_loading(rt::Runtime&r,bool){r.assertOwnerThread();if(ready_||r.device()!=device_||r.isGraphRecording())throw std::logic_error("GLM attention prepare postload differs");r.syncStream();ready_=true;}
AttentionPrepare::Output AttentionPrepare::forward(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&positions)const{
 return prepare_impl(r,x,positions,true);
}
AttentionPrepare::Output AttentionPrepare::forward_latent(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&positions)const{
 return prepare_impl(r,x,positions,false);
}
AttentionPrepare::Output AttentionPrepare::prepare_impl(rt::Runtime&r,const rt::Tensor&x,const rt::Tensor&positions,bool expand)const{
 r.assertOwnerThread();if(!ready_||r.device()!=device_||r.isGraphRecording()||!x||x->device()!=device_||x->dtype()!=rt::DataType::kBFloat16||x->ndim()!=2||!x->size(0)||x->size(1)!=6144||!positions||positions->device()!=device_||positions->dtype()!=rt::DataType::kInt64||positions->shape()!=rt::Shape{x->size(0)}||!positions->is_contiguous())throw std::invalid_argument("GLM attention prepare input/position contract differs");
 size_t tokens=x->size(0);std::vector<int64_t>host_positions(tokens);r.memcpyD2H(host_positions.data(),positions->data(),tokens*8);for(auto p:host_positions)if(p<0||size_t(p)>=maximum_)throw std::invalid_argument("GLM attention position out of bounds");
 Work work{r};
 // The checkpoint reference uses the attention RMSNorm default epsilon and
 // rounds the normalized activation to BF16 before multiplying its weight.
 auto norm=[&](const rt::Tensor&v,const rt::Tensor&w,const rt::Tensor&one){auto normalized=work.alloc(v->shape(),rt::DataType::kBFloat16),out=work.alloc(v->shape(),rt::DataType::kBFloat16);ex::submit<infini::ops::RmsNorm>(r,infini::ops::RmsNorm::DefaultConfig(device_.type()),v,one,float{1e-6f},normalized);ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),normalized,w,out);return out;};
 auto qa=qa_->forward(r,x);work.held.push_back(qa);auto resid=norm(qa,qn_,qones_);work.held.push_back(resid);auto query=qb_->forward(r,resid)->view({tokens,heads_,256});work.held.push_back(query);
 auto ka=ka_->forward(r,x);work.held.push_back(ka);auto latent=norm(ka->narrow({{1,0,512}}),kn_,kones_);work.held.push_back(latent);auto key_rot=ka->narrow({{1,512,64}})->contiguous(r)->view({tokens,1,64});work.held.push_back(key_rot);
 auto query_rot=query->narrow({{2,192,64}})->contiguous(r);work.held.push_back(query_rot);auto rotated=indexer_rotary(r,*rope_,query_rot,key_rot,positions,64);work.held.push_back(rotated.first);work.held.push_back(rotated.second);query->narrow({{2,192,64}})->copy_from(r,rotated.first);
 if(!expand){r.syncStream();return {query,{},{},resid,latent,rotated.second};}
 auto expanded=kb_->forward(r,latent)->view({tokens,heads_,448});work.held.push_back(expanded);auto key=work.alloc({tokens,heads_,256},rt::DataType::kBFloat16);key->narrow({{2,0,192}})->copy_from(r,expanded->narrow({{2,0,192}}));
 // Broadcast the single rotary key explicitly, preserving Runtime tensor ownership.
 for(size_t head=0;head<heads_;++head)key->narrow({{1,head,1},{2,192,64}})->copy_from(r,rotated.second);
 auto value=expanded->narrow({{2,192,256}})->contiguous(r);work.held.push_back(value);r.syncStream();return {query,key,value,resid,latent,rotated.second};
}
std::pair<rt::Tensor,rt::Tensor> AttentionPrepare::expand_cached(rt::Runtime&r,const rt::Tensor&latent,const rt::Tensor&rotary)const{
 r.assertOwnerThread();if(!ready_||r.device()!=device_||r.isGraphRecording()||!latent||latent->device()!=device_||latent->dtype()!=rt::DataType::kBFloat16||latent->ndim()!=2||!latent->size(0)||latent->size(1)!=512||!latent->is_contiguous()||!rotary||rotary->device()!=device_||rotary->dtype()!=latent->dtype()||rotary->shape()!=rt::Shape{latent->size(0),1,64})throw std::invalid_argument("GLM selected latent/rotary cache geometry differs");
 Work work{r};size_t tokens=latent->size(0);auto expanded=kb_->forward(r,latent)->view({tokens,heads_,448});work.held.push_back(expanded);auto key=work.alloc({tokens,heads_,256},rt::DataType::kBFloat16);key->narrow({{2,0,192}})->copy_from(r,expanded->narrow({{2,0,192}}));
 for(size_t head=0;head<heads_;++head)key->narrow({{1,head,1},{2,192,64}})->copy_from(r,rotary);
 auto value=expanded->narrow({{2,192,256}})->contiguous(r);work.held.push_back(value);r.syncStream();return {key,value};
}
}
