#include "models/infinilm_model.hpp"
#include "models/models_registry.hpp"
#include "global_state/forward_context.hpp"
#include "runtime/runtime.hpp"
#include "native_sparse_cache_r22.hpp"
#include "native_sparse_requests_r22.hpp"
#include "deepseek_v4_routed_text_model_r20.hpp"
#include "glm_moe_dsa_routed_text_model_r19.hpp"
#include <cstring>

namespace infinilm::models::native_sparse {
namespace rt=infinilm::runtime;
std::shared_ptr<config::ModelConfig>validate_config(std::shared_ptr<config::ModelConfig>cfg){
 const auto&j=cfg->get_config_json();const auto type=j.at("model_type").get<std::string>();
 if(cfg->get_dtype()!=rt::DataType::kBFloat16||cfg->get_quant_scheme()!=quantization::QuantScheme::MODEL_NATIVE||cfg->get_kv_quant_scheme()!=quantization::KVQuantAlgo::NONE)throw std::invalid_argument("Native sparse requires BF16 activations, owned checkpoint quantization and unquantized KV");
 if(type=="deepseek_v4"){
  if(j.at("num_hidden_layers")!=43||j.at("hidden_size")!=4096||j.at("hc_mult")!=4||j.at("vocab_size")!=129280||j.at("n_routed_experts")!=256||j.at("num_experts_per_tok")!=6)throw std::invalid_argument("Native DeepSeek requires qualified actual43-layer Flash channel-INT8 geometry");
 }else if(type=="glm_moe_dsa"){
  if(j.at("num_hidden_layers")!=78||j.at("hidden_size")!=6144||j.at("vocab_size")!=154880||j.at("n_routed_experts")!=256)throw std::invalid_argument("Native GLM requires qualified actual78-layer FP8 geometry");
 }else throw std::invalid_argument("Unknown native sparse family");
 return cfg;
}
template<typename T>std::vector<T>host_values(rt::Runtime&r,const std::optional<rt::Tensor>&value,rt::DataType dtype){
 if(!value||!value.value()||value.value()->dtype()!=dtype||value.value()->ndim()!=1||!value.value()->is_contiguous())throw std::invalid_argument("Native sparse requires explicit contiguous request metadata");
 const auto&t=value.value();std::vector<T>out(t->numel());
 if(t->device().type()==rt::Device::Type::kCpu)std::memcpy(out.data(),t->data(),t->nbytes());
 else{if(t->device()!=r.device())throw std::invalid_argument("Native sparse metadata device differs");r.syncStream();r.memcpyD2H(out.data(),t->data(),t->nbytes());}
 return out;
}
class NativeSparseModel final:public InfinilmModel {
public:
 NativeSparseModel(rt::Runtime&r,std::shared_ptr<config::ModelConfig>cfg,const engine::distributed::RankInfo&rank,backends::AttentionBackend backend,const layers::moe::EPConfig&ep):owner_(&r),rank_(rank){
  r.assertOwnerThread();validate_config(cfg);model_config_=std::move(cfg);const bool deep=model_config_->get<std::string>("model_type")=="deepseek_v4";
  if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rank.device!=r.device()||rank.tp_size!=8||rank.tp_rank<0||rank.tp_rank>=8||!rank.comm||rank.world_size!=rank.tp_size*rank.pp_size||rank.world_rank!=rank.pp_stage*rank.tp_size+rank.tp_rank||ep.backend!=layers::moe::EPBackend::Disabled||backend!=backends::AttentionBackend::NATIVE_SPARSE||(deep?(rank.pp_size!=1||rank.pp_stage!=0):(rank.pp_size!=4||rank.pp_stage<0||rank.pp_stage>=4||!rank.world_comm)))throw std::invalid_argument("Native sparse requires Hygon TP8, disabled EP, explicit native backend and qualified DeepSeekPP1/GLMPP4 topology");
 }
 void reset_cache(const cache::CacheConfig*base,global_state::ForwardContext&fc,backends::AttentionBackend backend,const engine::distributed::RankInfo&rank)override{
  auto&r=fc.require_runtime();check_owner(r);const auto*c=dynamic_cast<const cache::NativeSparseCacheConfig*>(base);
  if(!c||backend!=backends::AttentionBackend::NATIVE_SPARSE||rank.device!=rank_.device||rank.tp_size!=rank_.tp_size||rank.tp_rank!=rank_.tp_rank||rank.pp_size!=rank_.pp_size||rank.pp_stage!=rank_.pp_stage)throw std::invalid_argument("Native sparse cache/rank contract differs");
  if(deep_||glm_){if(c->rows()!=rows_||c->capacity()!=capacity_)throw std::invalid_argument("Changing native cache geometry requires a new loaded engine");r.syncStream();r.trimMemory();if(deep_)deep_->reset(r);else glm_->reset(r);}
  else{
   rows_=c->rows();capacity_=c->capacity();const auto&cfg=model_config_->get_config_json();
   if(cfg.at("model_type")=="deepseek_v4"){deep_=std::make_unique<deepseek_v4::TextModelR20>(r,cfg,rank_,rows_,capacity_);local_weights_=deep_->state_dict();}
   else{glm_=std::make_unique<glm_moe_dsa::TextModelR19>(r,cfg,rank_,rows_,capacity_);local_weights_=glm_->state_dict();}
  }
  lengths_.assign(rows_,0);cache_config_=c->unique_copy();fc.kv_cache_vec.clear();fc.conv_state_vec.clear();fc.ssm_state_vec.clear();r.syncStream();r.trimMemory();
 }
 void process_weights_after_loading(rt::Runtime&r,bool transpose)override{check_owner(r);if(transpose||ready_||(!deep_&&!glm_))throw std::invalid_argument("Native sparse postload state/transpose differs");if(deep_)deep_->process_weights_after_loading(r,false);else glm_->process_weights_after_loading(r,false);r.syncStream();ready_=true;}
 Output forward(const Input&input,global_state::ForwardContext&fc)const override{
  auto&r=fc.require_runtime();check_owner(r);
  if(!ready_||!input.input_ids||!input.position_ids||input.target_hidden_states||input.pixel_values||input.image_grid_thw)throw std::invalid_argument("Native sparse requires finalized text input");
  auto ids=input.input_ids.value(),positions=input.position_ids.value();
  if(ids->dtype()!=rt::DataType::kInt64||positions->dtype()!=rt::DataType::kInt64||ids->device()!=r.device()||positions->device()!=r.device()||!ids->is_contiguous()||!positions->is_contiguous()||ids->numel()!=positions->numel())throw std::invalid_argument("Native sparse packed IDs/positions differ");
  auto rows=host_values<int32_t>(r,input.mamba_init_state_indices,rt::DataType::kInt32),writes=host_values<int32_t>(r,input.mamba_final_state_indices,rt::DataType::kInt32);
  if(rows!=writes)throw std::invalid_argument("Native sparse requires same read/write physical state rows");
  auto past=host_values<int32_t>(r,input.past_sequence_lengths,rt::DataType::kInt32),total=host_values<int32_t>(r,input.total_sequence_lengths,rt::DataType::kInt32),offsets=host_values<int32_t>(r,input.input_offsets,rt::DataType::kInt32);
  auto requests=validate_requests(rows_,capacity_,lengths_,rows,past,total,offsets,ids->numel());
  rt::Tensor logits,hidden;
  if(deep_){std::vector<deepseek_v4::TextModelR20::Request>native;for(auto q:requests)native.push_back({q.row,q.past,q.tokens});auto out=deep_->forward(r,ids->view({ids->numel()}),positions->view({positions->numel()}),native,input.sample_all_positions);logits=out.head.logits;hidden=out.residual;}
  else{std::vector<glm_moe_dsa::NativeAttention::Request>native;for(auto q:requests)native.push_back({q.row,q.past,q.tokens});auto out=glm_->forward(r,ids->view({ids->numel()}),positions->view({positions->numel()}),native,input.sample_all_positions);logits=out.logits;hidden=out.hidden;}
  if(logits)logits=logits->view({1,input.sample_all_positions?ids->numel():requests.size(),size_t(model_config_->get<size_t>("vocab_size"))});
  commit_lengths(lengths_,requests);r.syncStream();return {logits,hidden};
 }
private:
 void check_owner(rt::Runtime&r)const{r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::invalid_argument("Native sparse Runtime owner/recording differs");}
 rt::Runtime*owner_;engine::distributed::RankInfo rank_;size_t rows_=0,capacity_=0;bool ready_=false;
 mutable std::vector<size_t>lengths_;std::unique_ptr<deepseek_v4::TextModelR20>deep_;std::unique_ptr<glm_moe_dsa::TextModelR19>glm_;
};
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(deepseek_v4,NativeSparseModel,validate_config);
INFINILM_REGISTER_EXPLICIT_CAUSAL_LM_MODEL(glm_moe_dsa,NativeSparseModel,validate_config);
}
