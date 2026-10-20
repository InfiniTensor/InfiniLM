#include "engine/infer_engine.hpp"
#include "config/config_factory.hpp"
#include "native_sparse_cache_r22.hpp"
#include <cuda_runtime_api.h>
#include <cstring>
#include <memory>
#include <set>
namespace rt=infinilm::runtime;namespace en=infinilm::engine;
namespace{thread_local std::string error,serialized;
struct ProbeEngine:en::InferEngine{using en::InferEngine::InferEngine;std::vector<en::RankWorker::Output>rank_outputs(){std::vector<en::RankWorker::Output>v;for(auto&w:workers_)v.push_back(w->get_output());return v;}};
struct Context{std::unique_ptr<ProbeEngine>engine;std::unique_ptr<infinilm::cache::NativeSparseCacheConfig>cache;size_t vocab=0;};
rt::Tensor cpu(void*p,rt::Shape shape,rt::DataType dtype){return rt::Tensor::from_blob(p,shape,dtype,{rt::Device::Type::kCpu,0});}
void copy_cpu(void*out,const rt::Tensor&x){if(!out||!x)throw std::invalid_argument("Native Engine output missing");if(x->device().type()==rt::Device::Type::kCpu)std::memcpy(out,x->data(),x->nbytes());else{int old=0;cudaGetDevice(&old);cudaSetDevice(x->device().index());auto result=cudaMemcpy(out,x->data(),x->nbytes(),cudaMemcpyDeviceToHost);cudaSetDevice(old);if(result!=cudaSuccess)throw std::runtime_error("Native Engine completed-output copy failed");}}
}
extern "C" const char*modern_native_engine_error(){return error.c_str();}
extern "C" int modern_native_engine_config(const char*raw){try{auto cfg=infinilm::config::ConfigFactory::createConfig(raw);if(cfg->get_quant_scheme()!=infinilm::quantization::QuantScheme::MODEL_NATIVE)throw std::invalid_argument("Expected typed model-owned quantization");return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" void*modern_native_engine_create(const char*raw,size_t rows,size_t capacity){try{
 auto c=std::make_unique<Context>();auto cfg=nlohmann::json::parse(raw);if(cfg.at("model_type")!="deepseek_v4")throw std::invalid_argument("Initial Engine qualification requires DeepSeekTP8PP1; GLM factory qualification follows separately");c->vocab=cfg.at("vocab_size");c->cache=std::make_unique<infinilm::cache::NativeSparseCacheConfig>(rows,capacity);
 en::distributed::DistConfig dist(8);c->engine=std::make_unique<ProbeEngine>(cfg.dump(),dist,rt::Device::Type::kHygon,c->cache.get(),false,infinilm::backends::AttentionBackend::NATIVE_SPARSE,std::nullopt,false,"async",false);return c.release();
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" const char*modern_native_engine_keys(void*raw){try{serialized=nlohmann::json(static_cast<Context*>(raw)->engine->state_dict_keys()).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_native_engine_load(void*raw,const char*descriptors){try{
 auto*c=static_cast<Context*>(raw);std::unordered_map<std::string,rt::Tensor>sources;
 for(auto&v:nlohmann::json::parse(descriptors)){std::string name=v.at("name");int type=v.at("dtype");auto p=reinterpret_cast<void*>(v.at("pointer").get<uintptr_t>());if(!p||type<0||type>3||sources.count(name))throw std::invalid_argument("Native Engine load descriptor differs");sources.emplace(name,cpu(p,v.at("shape").get<rt::Shape>(),type==0?rt::DataType::kFloat32:type==1?rt::DataType::kBFloat16:type==2?rt::DataType::kInt8:rt::DataType::kInt64));}
 c->engine->load_params(sources,true);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_native_engine_finish(void*raw){try{static_cast<Context*>(raw)->engine->process_weights_after_loading();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_native_engine_forward(void*raw,size_t tokens,size_t count,const int32_t*rows,const int32_t*past,const int32_t*total,const int32_t*offsets,const int64_t*ids,const int64_t*positions,void*out_logits,void*out_ids){try{
 auto*c=static_cast<Context*>(raw);if(!tokens||tokens>4096||!count||count>c->cache->rows()||!rows||!past||!total||!offsets||!ids||!positions||!out_logits||!out_ids)throw std::invalid_argument("Native Engine CAPI input differs");
 en::InferEngine::Input input;input.input_ids=cpu(const_cast<int64_t*>(ids),{1,tokens},rt::DataType::kInt64);input.position_ids=cpu(const_cast<int64_t*>(positions),{1,tokens},rt::DataType::kInt64);input.past_sequence_lengths=cpu(const_cast<int32_t*>(past),{count},rt::DataType::kInt32);input.total_sequence_lengths=cpu(const_cast<int32_t*>(total),{count},rt::DataType::kInt32);input.input_offsets=cpu(const_cast<int32_t*>(offsets),{count+1},rt::DataType::kInt32);input.mamba_init_state_indices=cpu(const_cast<int32_t*>(rows),{count},rt::DataType::kInt32);input.mamba_final_state_indices=input.mamba_init_state_indices;input.top_k=1;input.top_p=1;input.temperature=1;
 auto output=c->engine->forward(input);if(!output.logits||output.logits->numel()!=count*c->vocab||output.logits->dtype()!=rt::DataType::kBFloat16||!output.output_ids||output.output_ids->numel()!=count)throw std::runtime_error("Native Engine sampled/logits output differs");copy_cpu(out_logits,output.logits);
 if(output.output_ids->dtype()==rt::DataType::kInt64)copy_cpu(out_ids,output.output_ids);else throw std::runtime_error("Native Engine output ID dtype differs");return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_native_engine_reset(void*raw){try{auto*c=static_cast<Context*>(raw);c->engine->reset_cache(c->cache.get());return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_native_engine_rank_logits_equal(void*raw){try{auto*c=static_cast<Context*>(raw);auto outputs=c->engine->rank_outputs();if(outputs.size()!=8||!outputs[0].logits)throw std::runtime_error("Native Engine rank logits missing");std::vector<std::byte>reference(outputs[0].logits->nbytes()),actual(reference.size());copy_cpu(reference.data(),outputs[0].logits);for(size_t i=1;i<outputs.size();++i){if(!outputs[i].logits||outputs[i].logits->nbytes()!=reference.size())throw std::runtime_error("Native Engine rank logits shape differs");copy_cpu(actual.data(),outputs[i].logits);if(actual!=reference)throw std::runtime_error("Native Engine TP full logits differ");}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_native_engine_destroy(void*raw){try{delete static_cast<Context*>(raw);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
