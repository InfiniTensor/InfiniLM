#include "glm_fp8_runtime_r809.hpp"
#include "glm_moe_dsa_projected_text_model_r22.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <cstring>
#include <map>
#include <set>
#include <unordered_map>
namespace rt=infinilm::runtime;namespace model=infinilm::models::glm_moe_dsa;
namespace{
thread_local std::string error,serialized;
struct Context{std::shared_ptr<rt::Runtime>owner;std::unique_ptr<model::TextModelProjectedR22>module;infinicclComm_t world=nullptr,tp=nullptr,control=nullptr;std::set<std::string>loaded;int stage=0;bool ready=false;};
std::map<int,std::shared_ptr<rt::Runtime>>&owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>pool;return pool;}
void drain(Context*c){if(c&&c->owner)try{c->owner->syncStream();}catch(...){std::terminate();}}
void producer(void*s){if(cudaStreamSynchronize(static_cast<cudaStream_t>(s))!=cudaSuccess)throw std::runtime_error("GLM full-model producer failed");}
}
extern "C" const char*modern_glm_model_error(){return error.c_str();}
extern "C" size_t modern_glm_model_id_size(){return sizeof(infinicclUniqueId);}
extern "C" int modern_glm_model_id(void*out){infinicclUniqueId id{};if(infinicclGetUniqueId(&id)!=infinicclSuccess)return 1;std::memcpy(out,&id,sizeof(id));return 0;}
extern "C" void*modern_glm_model_create(int device,int stage,size_t rows,size_t capacity,const char*cfg,const void*world_id,const void*tp_id,const void*control_id){std::unique_ptr<Context>c;try{
 if(device<0||device>=8||stage<0||stage>=4)throw std::invalid_argument("GLM full-model requires TP8 PP4");c=std::make_unique<Context>();auto&p=owners()[device];if(!p){p=rt::Runtime::create({rt::Device::Type::kHygon,device});p->enableOwnerDrainFailStop();}c->owner=p;c->stage=stage;auto&r=*p;infini::rt::DeviceGuard guard{r.device().type(),device,true};
 infinicclUniqueId world{},tp{},control{};std::memcpy(&world,world_id,sizeof(world));std::memcpy(&tp,tp_id,sizeof(tp));std::memcpy(&control,control_id,sizeof(control));
 if(infinicclCommInitRank(&c->world,32,world,stage*8+device)!=infinicclSuccess)throw std::runtime_error("GLM world init failed");
 if(infinicclCommInitRank(&c->tp,8,tp,device)!=infinicclSuccess)throw std::runtime_error("GLM TP init failed");
 if(infinicclCommInitRank(&c->control,32,control,stage*8+device)!=infinicclSuccess)throw std::runtime_error("GLM control init failed");
 infinilm::engine::distributed::RankInfo rank(r.device());rank.tp_size=8;rank.tp_rank=device;rank.comm=c->tp;rank.pp_size=4;rank.pp_stage=stage;rank.world_size=32;rank.world_rank=stage*8+device;rank.world_comm=c->world;
 c->module=std::make_unique<model::TextModelProjectedR22>(r,nlohmann::json::parse(cfg),rank,rows,capacity);r.syncStream();return c.release();
 }catch(const std::exception&e){error=e.what();drain(c.get());return nullptr;}}
extern "C" const char*modern_glm_model_keys(void*raw){serialized=nlohmann::json(static_cast<Context*>(raw)->module->state_dict_keys()).dump();return serialized.c_str();}
extern "C" int modern_glm_model_load_batch(void*raw,const char*descriptors){auto*c=static_cast<Context*>(raw);try{
 if(c->ready)throw std::logic_error("GLM model weights already finalized");auto values=nlohmann::json::parse(descriptors);std::unordered_map<std::string,rt::Tensor>sources;
 for(auto&item:values){std::string name=item.at("name");int type=item.at("dtype");auto shape=item.at("shape").get<rt::Shape>();auto p=reinterpret_cast<void*>(item.at("pointer").get<uintptr_t>());if(type<0||type>2||!p||sources.count(name))throw std::invalid_argument("GLM batch source differs");sources.emplace(name,rt::Tensor::from_blob(p,shape,type==0?rt::DataType::kFloat32:type==1?rt::DataType::kBFloat16:rt::DataType::kUInt8,{rt::Device::Type::kCpu,0}));}
 c->module->load_parameters_no_sync(*c->owner,sources,true);c->owner->syncStream();for(auto&item:sources)c->loaded.insert(item.first);return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_finish(void*raw){auto*c=static_cast<Context*>(raw);try{if(c->ready)throw std::logic_error("GLM model already finalized");for(auto&name:c->module->state_dict_keys())if(!c->loaded.count(name))throw std::runtime_error("Missing GLM model checkpoint parameter "+name);c->module->process_weights_after_loading(*c->owner,false);c->owner->syncStream();c->ready=true;return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_forward(void*raw,size_t tokens,size_t count,const size_t*requests,const void*input,const void*positions,void*logits,int all_positions,void*stream){auto*c=static_cast<Context*>(raw);try{
 if(!c->ready||!count||count>128||!requests)throw std::invalid_argument("GLM full-model CAPI metadata differs");producer(stream);auto&r=*c->owner;std::vector<model::NativeAttentionProjectedR22::Request>metadata;for(size_t i=0;i<count;++i)metadata.push_back({requests[3*i],requests[3*i+1],requests[3*i+2]});
 auto ids=rt::Tensor::from_blob(const_cast<void*>(input),{tokens},rt::DataType::kInt64,r.device()),pos=rt::Tensor::from_blob(const_cast<void*>(positions),{tokens},rt::DataType::kInt64,r.device());auto result=c->module->forward(r,ids,pos,metadata,all_positions!=0);
 if(c->stage==3){if(!result.logits)throw std::runtime_error("Missing final-stage GLM logits");rt::Tensor::from_blob(logits,result.logits->shape(),result.logits->dtype(),r.device())->copy_from(r,result.logits);}r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_control(void*raw,size_t count,void*values,void*stream){auto*c=static_cast<Context*>(raw);try{producer(stream);auto&r=*c->owner;auto x=rt::Tensor::from_blob(values,{count},rt::DataType::kFloat32,r.device());infinilm::engine::distributed::allreduce_(r,x,x,infinicclSum,c->control);r.syncStream();return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_reset(void*raw){auto*c=static_cast<Context*>(raw);try{c->module->reset(*c->owner);return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
extern "C" int modern_glm_model_destroy(void*raw){auto*c=static_cast<Context*>(raw);drain(c);if(modern_glm_fp8_runtime_clear_r809(c->owner.get())){error=modern_glm_fp8_runtime_error_r809();return 1;}c->module.reset();drain(c);if(infinicclCommDestroy(c->control)!=infinicclSuccess||infinicclCommDestroy(c->tp)!=infinicclSuccess||infinicclCommDestroy(c->world)!=infinicclSuccess)return 1;delete c;return 0;}
extern "C" int modern_glm_model_shutdown(){try{for(auto&item:owners()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second.reset();}owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}

// Experimental control is local to one TP stage. Numerical model code is unchanged.
extern "C" int modern_glm_model_tp_control(void*raw,size_t count,void*values,void*stream){auto*c=static_cast<Context*>(raw);try{producer(stream);auto&r=*c->owner;auto x=rt::Tensor::from_blob(values,{count},rt::DataType::kFloat32,r.device());infinilm::engine::distributed::allreduce_(r,x,x,infinicclSum,c->tp);r.syncStream();return 0;}catch(const std::exception&e){error=e.what();drain(c);return 1;}}
