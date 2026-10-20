#include "kimi_engine_input_r145.hpp"
#include "models/kimi_k3/kimi_k3_pipeline_partition.hpp"
#include "engine/infer_engine.hpp"
#include "config/config_factory.hpp"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <cstring>
#include <memory>
#include <mutex>
#include <vector>
#include <chrono>
namespace rt=infinilm::runtime;namespace en=infinilm::engine;
namespace{
thread_local std::string error,serialized;
struct ProbeEngine:en::InferEngine{using en::InferEngine::InferEngine;std::vector<en::RankWorker::Output>rank_outputs(){std::vector<en::RankWorker::Output>v;for(auto&w:workers_)v.push_back(w->get_output());return v;}};
struct OutputMetricsR963{uint64_t engine_calls=0,full_logits_calls=0,token_only_calls=0,forward_logits_bytes=0,forward_id_bytes=0;double engine_wall_seconds=0,logits_copy_seconds=0,id_copy_seconds=0;};
struct Context{bool speculation_r3632=false;std::unique_ptr<ProbeEngine>engine;std::unique_ptr<infinilm::cache::PagedKVCacheConfig>cache;std::mutex mutex;OutputMetricsR963 metrics;size_t vocab=0,max_rows=0;int stage=0,pp_size=4,tp_nodes=1,tp_node_rank=0,node=0;};
rt::Tensor cpu(void*p,rt::Shape shape,rt::DataType dtype){return rt::Tensor::from_blob(p,shape,dtype,{rt::Device::Type::kCpu,0});}
void copy_cpu(void*out,const rt::Tensor&x){if(!out||!x)throw std::invalid_argument("Kimi Engine output missing");if(x->device().type()==rt::Device::Type::kCpu){std::memcpy(out,x->data(),x->nbytes());return;}int old=0;auto code=cudaGetDevice(&old);if(code!=cudaSuccess)throw std::runtime_error("Get current device");code=cudaSetDevice(x->device().index());if(code!=cudaSuccess)throw std::runtime_error("Select completed output device");code=cudaMemcpy(out,x->data(),x->nbytes(),cudaMemcpyDeviceToHost);auto restored=cudaSetDevice(old);if(code!=cudaSuccess||restored!=cudaSuccess)throw std::runtime_error("Kimi Engine completed output copy failed");}
}
extern "C" const char*modern_kimi_engine_error_r2475(){return error.c_str();}
extern "C" void*modern_kimi_engine_create_r2475(const char*raw,int node,int tp_nodes,int pp_size,const char*address,int port,size_t max_rows){try{
 if((max_rows<1||max_rows>128)||!raw||!address||(tp_nodes!=1&&tp_nodes!=2&&tp_nodes!=4&&tp_nodes!=6)||pp_size<1||pp_size>4||(tp_nodes==1&&pp_size!=4)||(tp_nodes==6&&pp_size!=1)||node<0||node>=tp_nodes*pp_size||port<=0||port>65535)throw std::invalid_argument("Kimi Engine bootstrap");
 const int stage=node/tp_nodes;
 auto cfg=nlohmann::json::parse(raw);auto&text=cfg.at("text_config");auto quant=text.at("quantization_config");auto w=quant.at("config_groups").at("group_0").at("weights");
 if(cfg.at("model_type")!="kimi_k3"||text.at("num_hidden_layers")!=93||text.at("hidden_size")!=7168||quant.at("format")!="mxfp4-pack-quantized"||quant.at("quant_method")!="compressed-tensors"||w.at("group_size")!=32||w.at("num_bits")!=4||w.at("scale_dtype")!="torch.uint8"||w.at("type")!="float"||!w.at("symmetric").get<bool>()||w.at("dynamic").get<bool>())throw std::invalid_argument("Kimi native checkpoint geometry/MXFP4 metadata differs");
 cfg["infinilm_text_only"]=true;cfg["infinilm_kimi_global_tp_adapter"]=true;text["infinilm_kimi_global_tp_adapter"]=true;cfg["infinilm_kimi_engine_adapter"]=true;cfg["dtype"]="bfloat16";cfg["quantization_config"]=nullptr;cfg.erase("lm_head_output_dtype");text["dtype"]="bfloat16";text.erase("lm_head_output_dtype");
 auto c=std::make_unique<Context>();c->stage=stage;c->pp_size=pp_size;c->tp_nodes=tp_nodes;c->tp_node_rank=node%tp_nodes;c->node=node;c->max_rows=max_rows;c->vocab=text.at("vocab_size");c->cache=std::make_unique<infinilm::cache::PagedKVCacheConfig>(4*max_rows+4,256,max_rows);en::distributed::DistConfig dist(8);dist.pp_size=pp_size;dist.pp_stage=stage;dist.tp_node_size=tp_nodes;dist.tp_node_rank=node%tp_nodes;dist.master_addr=address;dist.master_port=port;
 c->engine=std::make_unique<ProbeEngine>(cfg.dump(),dist,rt::Device::Type::kHygon,c->cache.get(),false,infinilm::backends::AttentionBackend::FLASH_ATTN,std::nullopt,false,"async",false);return c.release();
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" const char*modern_kimi_engine_keys_r2475(void*raw){try{auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);serialized=nlohmann::json(c->engine->state_dict_keys()).dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int modern_kimi_engine_load_r2475(void*raw,const char*descriptors){try{
 auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);std::unordered_map<std::string,rt::Tensor>sources;
 for(const auto&v:nlohmann::json::parse(descriptors)){std::string name=v.at("name");int type=v.at("dtype");auto p=reinterpret_cast<void*>(v.at("pointer").get<uintptr_t>());if(!p||type<0||type>4||sources.count(name))throw std::invalid_argument("Kimi weight descriptor");auto dtype=type==0?rt::DataType::kFloat32:type==1?rt::DataType::kBFloat16:type==2?rt::DataType::kUInt8:type==3?rt::DataType::kInt64:rt::DataType::kInt8;sources.emplace(name,cpu(p,v.at("shape").get<rt::Shape>(),dtype));}
 c->engine->load_params(sources,true);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_kimi_engine_finish_r2475(void*raw){try{auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);c->engine->process_weights_after_loading();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_kimi_engine_forward_r2475(void*raw,size_t tokens,size_t count,int flags,const int32_t*rows,const int32_t*past,const int32_t*total,const int32_t*offsets,const int64_t*ids,const int64_t*positions,void*out_logits,void*out_ids,int*out_flags){try{
 auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);if(c->speculation_r3632){if(!rows||!count||count>c->max_rows/2)throw std::invalid_argument("3632 normal prefill count");for(size_t i=0;i<count;++i)if(rows[i]<1||size_t(rows[i])>c->max_rows/2||past[i]!=0)throw std::invalid_argument("3632 normal API is initial prefill only; subsequent decode uses paired-state API");}if(!tokens||tokens>8192||!count||count>c->max_rows||flags<0||(flags&~119)||((flags&32)&&!(flags&16))||!rows||!past||!total||!offsets||!ids||!positions||!out_flags||(c->stage+1==c->pp_size&&((!(flags&4)&&!out_logits)||!out_ids)))throw std::invalid_argument("Kimi Engine CAPI input");
 if((flags&64)&&c->tp_nodes!=1)throw std::invalid_argument("2347 TP8-specific tile is forbidden on wider TP");
 std::vector<int32_t>cu(count+1,0),table(count*4);std::vector<int64_t>slots(tokens);
 if(offsets[0]!=0||offsets[count]<0||size_t(offsets[count])!=tokens)throw std::invalid_argument("Kimi offsets");
 for(size_t i=0;i<count;++i){if(rows[i]<1||size_t(rows[i])>c->max_rows||past[i]<0||total[i]<=past[i]||total[i]>1024||offsets[i]<0||offsets[i+1]<=offsets[i]||int64_t(offsets[i+1])-offsets[i]!=int64_t(total[i])-past[i])throw std::invalid_argument("Kimi row/length geometry");cu[i+1]=cu[i]+total[i];int32_t page=1+(rows[i]-1)*4;for(size_t j=0;j<4;++j)table[i*4+j]=page+int32_t(j);for(int32_t j=offsets[i];j<offsets[i+1];++j){auto position=int64_t(past[i])+j-offsets[i];slots[size_t(j)]=(page+position/256)*256+position%256;}}
 en::InferEngine::Input a;a.input_ids=cpu(const_cast<int64_t*>(ids),{1,tokens},rt::DataType::kInt64);a.position_ids=cpu(const_cast<int64_t*>(positions),{1,tokens},rt::DataType::kInt64);a.past_sequence_lengths=cpu(const_cast<int32_t*>(past),{count},rt::DataType::kInt32);a.total_sequence_lengths=cpu(const_cast<int32_t*>(total),{count},rt::DataType::kInt32);a.input_offsets=cpu(const_cast<int32_t*>(offsets),{count+1},rt::DataType::kInt32);a.cu_seqlens=cpu(cu.data(),{count+1},rt::DataType::kInt32);a.block_tables=cpu(table.data(),{count,4},rt::DataType::kInt32);a.slot_mapping=cpu(slots.data(),{tokens},rt::DataType::kInt64);a.mamba_init_state_indices=cpu(const_cast<int32_t*>(rows),{count},rt::DataType::kInt32);a.mamba_final_state_indices=a.mamba_init_state_indices;a.top_k=1;a.top_p=1;a.temperature=1;a.kimi_grouped_prefill=flags&1;a.kimi_tiled_w13_r1861=flags&64;a.kimi_owned_indices_r1213=flags&16;a.kimi_identity_indices_r1213=flags&32;a.defer_pipeline_feedback=flags&2;
 auto start963=std::chrono::steady_clock::now();auto result=c->engine->forward(a);c->metrics.engine_wall_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start963).count();++c->metrics.engine_calls;int published=0;
 if(c->stage+1==c->pp_size){if(!result.logits||result.logits->shape()!=rt::Shape{1,count,c->vocab}||result.logits->dtype()!=rt::DataType::kBFloat16)throw std::runtime_error("Kimi tail logits geometry");if(flags&4){++c->metrics.token_only_calls;}else{auto tick963=std::chrono::steady_clock::now();copy_cpu(out_logits,result.logits);c->metrics.logits_copy_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick963).count();c->metrics.forward_logits_bytes+=result.logits->nbytes();++c->metrics.full_logits_calls;published|=1;}}
 if(result.output_ids){if(!out_ids||result.output_ids->dtype()!=rt::DataType::kInt64||result.output_ids->numel()!=count)throw std::runtime_error("Kimi sampled output geometry");auto tick963=std::chrono::steady_clock::now();copy_cpu(out_ids,result.output_ids);c->metrics.id_copy_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick963).count();c->metrics.forward_id_bytes+=result.output_ids->nbytes();published|=2;}
 if(c->stage+1==c->pp_size&&published!=((flags&4?0:1)+(c->tp_node_rank==0?2:0)))throw std::runtime_error("Kimi tail did not publish greedy tokens");*out_flags=published;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_kimi_engine_rank_logits_r2475(void*raw){try{auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);if(c->stage+1!=c->pp_size)throw std::invalid_argument("Tail logits only");auto outputs=c->engine->rank_outputs();if(outputs.size()!=8||!outputs[0].logits)throw std::runtime_error("Kimi all8 tail logits missing");std::vector<std::byte>base(outputs[0].logits->nbytes()),actual(base.size());copy_cpu(base.data(),outputs[0].logits);for(size_t k=1;k<8;++k){if(!outputs[k].logits||outputs[k].logits->shape()!=outputs[0].logits->shape())throw std::runtime_error("Kimi all8 tail logits shape differs");copy_cpu(actual.data(),outputs[k].logits);if(actual!=base)throw std::runtime_error("Kimi all8 tail logits differ");}return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_kimi_engine_reset_r2475(void*raw){try{auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);c->engine->reset_cache(c->cache.get());return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_kimi_engine_destroy_r2475(void*raw){try{delete static_cast<Context*>(raw);return 0;}catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int modern_kimi_engine_partition_r2475(int stage,int pp_size,size_t*out){try{if(!out||pp_size<1||pp_size>4||stage<0||stage>=pp_size)throw std::invalid_argument("876 stage");auto range=infinilm::models::kimi_k3::kimi_k3_pipeline_layer_range(93,size_t(pp_size),size_t(stage));out[0]=range.first;out[1]=range.second;return 0;}catch(const std::exception&e){error=e.what();return 1;}}

extern "C" const char*modern_kimi_engine_output_metrics_r2475(void*raw){try{if(!raw)throw std::invalid_argument("963metrics context");auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);auto&m=c->metrics;serialized=nlohmann::json{{"stage",c->stage},{"engine_calls",m.engine_calls},{"full_logits_calls",m.full_logits_calls},{"token_only_calls",m.token_only_calls},{"forward_logits_bytes",m.forward_logits_bytes},{"forward_id_bytes",m.forward_id_bytes},{"engine_wall_seconds",m.engine_wall_seconds},{"logits_copy_seconds",m.logits_copy_seconds},{"id_copy_seconds",m.id_copy_seconds}}.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}

extern "C" int modern_kimi_engine_resize_cache_r2475(void*raw,size_t rows){try{
 if(!raw||(rows<1||rows>128))throw std::invalid_argument("1098 cache capacity must be within 1..128");
 auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);if(c->speculation_r3632)throw std::invalid_argument("3632 fixed capacity paired state context");
 auto next=std::make_unique<infinilm::cache::PagedKVCacheConfig>(4*rows+4,256,rows);
 c->engine->reset_cache(next.get());c->cache=std::move(next);c->max_rows=rows;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_engine_cache_geometry_r2475(void*raw){try{
 if(!raw)throw std::invalid_argument("1098 null context");auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);
 serialized=nlohmann::json{{"max_rows",c->max_rows},{"pages",c->cache->num_blocks()},{"page_size",c->cache->block_size()},{"KDA_pool_rows",c->cache->num_blocks()/4}}.dump();return serialized.c_str();
 }catch(const std::exception&e){error=e.what();return nullptr;}}

extern "C" const char*modern_kimi_engine_layout_r2475(void*raw){try{auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);serialized=nlohmann::json{{"node",c->node},{"local_workers",8},{"TP",8*c->tp_nodes},{"TP_node_rank",c->tp_node_rank},{"PP",c->pp_size},{"PP_stage",c->stage},{"sampling_node",(c->pp_size-1)*c->tp_nodes},{"public_graph_enabled",false},{"owned_metadata_enabled",false}}.dump();return serialized.c_str();}catch(const std::exception&e){error=e.what();return nullptr;}}

extern "C" int modern_kimi_engine_spec_r3632(void*raw,size_t tokens,size_t count,int flags,const int32_t*read_state,const int32_t*write_state,const int32_t*rows,const int32_t*past,const int32_t*total,const int32_t*offsets,const int64_t*ids,const int64_t*positions,void*out_logits,void*out_ids,int*out_flags){try{
 auto*c=static_cast<Context*>(raw);std::lock_guard<std::mutex>lock(c->mutex);if(!c->speculation_r3632||count>c->max_rows/2||!rows||!read_state||!write_state||!(flags&2))throw std::invalid_argument("3632 dedicated shadow context and deferred feedback required");if(!tokens||tokens>8192||!count||count>c->max_rows||flags<0||(flags&~7)||((flags&32)&&!(flags&16))||!rows||!past||!total||!offsets||!ids||!positions||!out_flags||(c->stage+1==c->pp_size&&((!(flags&4)&&!out_logits)||!out_ids)))throw std::invalid_argument("Kimi Engine CAPI input");
 if((flags&64)&&c->tp_nodes!=1)throw std::invalid_argument("2347 TP8-specific tile is forbidden on wider TP");
 std::vector<int32_t>cu(count+1,0),table(count*4);std::vector<int64_t>slots(tokens);
 if(offsets[0]!=0||offsets[count]<0||size_t(offsets[count])!=tokens)throw std::invalid_argument("Kimi offsets");
 for(size_t i=0;i<count;++i){if(rows[i]<1||size_t(rows[i])>c->max_rows||past[i]<0||total[i]<=past[i]||total[i]>1024||offsets[i]<0||offsets[i+1]<=offsets[i]||int64_t(offsets[i+1])-offsets[i]!=int64_t(total[i])-past[i])throw std::invalid_argument("Kimi row/length geometry");cu[i+1]=cu[i]+total[i];int32_t page=1+(rows[i]-1)*4;for(size_t j=0;j<4;++j)table[i*4+j]=page+int32_t(j);for(int32_t j=offsets[i];j<offsets[i+1];++j){auto position=int64_t(past[i])+j-offsets[i];slots[size_t(j)]=(page+position/256)*256+position%256;}}
 en::InferEngine::Input a;a.input_ids=cpu(const_cast<int64_t*>(ids),{1,tokens},rt::DataType::kInt64);a.position_ids=cpu(const_cast<int64_t*>(positions),{1,tokens},rt::DataType::kInt64);a.past_sequence_lengths=cpu(const_cast<int32_t*>(past),{count},rt::DataType::kInt32);a.total_sequence_lengths=cpu(const_cast<int32_t*>(total),{count},rt::DataType::kInt32);a.input_offsets=cpu(const_cast<int32_t*>(offsets),{count+1},rt::DataType::kInt32);a.cu_seqlens=cpu(cu.data(),{count+1},rt::DataType::kInt32);a.block_tables=cpu(table.data(),{count,4},rt::DataType::kInt32);a.slot_mapping=cpu(slots.data(),{tokens},rt::DataType::kInt64);a.mamba_init_state_indices=cpu(const_cast<int32_t*>(read_state),{count},rt::DataType::kInt32);a.mamba_final_state_indices=cpu(const_cast<int32_t*>(write_state),{count},rt::DataType::kInt32);a.sample_all_positions=true;a.top_k=1;a.top_p=1;a.temperature=1;a.kimi_grouped_prefill=flags&1;a.kimi_tiled_w13_r1861=flags&64;a.kimi_owned_indices_r1213=flags&16;a.kimi_identity_indices_r1213=flags&32;a.defer_pipeline_feedback=flags&2;
 infinilm::models::kimi_native::SpecScopeR3632 scope(static_cast<en::InferEngine*>(c->engine.get()),c->max_rows,{rows,rows+count},{read_state,read_state+count},{write_state,write_state+count});
 auto start963=std::chrono::steady_clock::now();auto result=c->engine->forward(a);c->metrics.engine_wall_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-start963).count();++c->metrics.engine_calls;int published=0;
 if(c->stage+1==c->pp_size){if(!result.logits||result.logits->shape()!=rt::Shape{1,tokens,c->vocab}||result.logits->dtype()!=rt::DataType::kBFloat16)throw std::runtime_error("Kimi tail logits geometry");if(flags&4){++c->metrics.token_only_calls;}else{auto tick963=std::chrono::steady_clock::now();copy_cpu(out_logits,result.logits);c->metrics.logits_copy_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick963).count();c->metrics.forward_logits_bytes+=result.logits->nbytes();++c->metrics.full_logits_calls;published|=1;}}
 if(result.output_ids){if(!out_ids||result.output_ids->dtype()!=rt::DataType::kInt64||result.output_ids->numel()!=tokens)throw std::runtime_error("Kimi sampled output geometry");auto tick963=std::chrono::steady_clock::now();copy_cpu(out_ids,result.output_ids);c->metrics.id_copy_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-tick963).count();c->metrics.forward_id_bytes+=result.output_ids->nbytes();published|=2;}
 if(c->stage+1==c->pp_size&&published!=((flags&4?0:1)+(c->tp_node_rank==0?2:0)))throw std::runtime_error("Kimi tail did not publish greedy tokens");*out_flags=published;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}

extern "C" void*modern_kimi_engine_create_spec_r3632(const char*raw,int node,int tp_nodes,int pp_size,const char*address,int port,size_t rows){try{if(!infinilm::models::kimi_native::spec_enabled_r3632()||rows<2||rows>128||rows%2||tp_nodes!=1||pp_size!=4)throw std::invalid_argument("3632 optin evenrows TP8PP4 required");auto*p=modern_kimi_engine_create_r2475(raw,node,tp_nodes,pp_size,address,port,rows);if(p)static_cast<Context*>(p)->speculation_r3632=true;return p;}catch(const std::exception&e){error=e.what();return nullptr;}}

#include <functional>
extern "C" int modern_kimi_spec_contract_r3632(){try{
 using namespace infinilm::models::kimi_native;
 size_t positive=0,negative=0;
 auto reject=[&](const std::function<void()>&f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}if(!caught)throw std::runtime_error("3632 invalid CPU contract accepted");++negative;};
 if(!spec_enabled_r3632())throw std::invalid_argument("3632 explicit test optin required");
 for(size_t B: {size_t(1),size_t(2),size_t(4),size_t(8),size_t(16),size_t(32),size_t(64)})for(size_t W=1;W<=8;++W){
  size_t cap=2*B,N=B*W;std::vector<size_t>lengths(cap+1,0);std::vector<int32_t>rows(B),read(B),write(B),past(B),total(B),offset(B+1),cu(B+1),table(B*4);std::vector<int64_t>ids(N,17),positions(N),slots(N);
  for(size_t i=0;i<B;++i){rows[i]=read[i]=int32_t(i+1);write[i]=int32_t(i+1+B);past[i]=int32_t(7+i);lengths[i+1]=size_t(past[i]);total[i]=past[i]+int32_t(W);offset[i+1]=int32_t((i+1)*W);cu[i+1]=cu[i]+total[i];int32_t page=1+int32_t(i)*4;for(size_t j=0;j<4;++j)table[i*4+j]=page+int32_t(j);for(size_t j=0;j<W;++j){positions[i*W+j]=past[i]+int64_t(j);slots[i*W+j]=(page+positions[i*W+j]/256)*256+positions[i*W+j]%256;}}
  en::InferEngine::Input a;a.input_ids=cpu(ids.data(),{1,N},rt::DataType::kInt64);a.position_ids=cpu(positions.data(),{1,N},rt::DataType::kInt64);a.past_sequence_lengths=cpu(past.data(),{B},rt::DataType::kInt32);a.total_sequence_lengths=cpu(total.data(),{B},rt::DataType::kInt32);a.input_offsets=cpu(offset.data(),{B+1},rt::DataType::kInt32);a.cu_seqlens=cpu(cu.data(),{B+1},rt::DataType::kInt32);a.block_tables=cpu(table.data(),{B,4},rt::DataType::kInt32);a.slot_mapping=cpu(slots.data(),{N},rt::DataType::kInt64);a.mamba_init_state_indices=cpu(read.data(),{B},rt::DataType::kInt32);a.mamba_final_state_indices=cpu(write.data(),{B},rt::DataType::kInt32);a.sample_all_positions=true;a.top_k=1;a.top_p=1;a.temperature=1;
  int owner=1;
  {SpecScopeR3632 scope(&owner,cap,rows,read,write);auto b=a;auto plan=freeze_input(b,lengths,163840,&owner);if(plan.size()!=B)throw std::runtime_error("3632 plan size");auto committed=lengths;commit(committed,plan);for(size_t i=0;i<B;++i)if(plan[i].row!=size_t(write[i])||committed[size_t(read[i])]!=lengths[size_t(read[i])]||committed[size_t(write[i])]!=size_t(total[i]))throw std::runtime_error("3632 shadow length commit");++positive;}
  if(B==2&&W==2){
   reject([&]{auto b=a;freeze_input(b,lengths,163840,&owner);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,read);});
   reject([&]{auto bad=write;std::swap(bad[0],bad[1]);SpecScopeR3632 s(&owner,cap,rows,read,bad);});
   reject([&]{auto bad=rows;bad[1]=bad[0];SpecScopeR3632 s(&owner,cap,bad,read,write);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);SpecScopeR3632 t(&owner,cap,rows,read,write);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;freeze_input(b,lengths,163840,nullptr);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;b.sample_all_positions=false;freeze_input(b,lengths,163840,&owner);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto bad=lengths;bad[1]++;auto b=a;freeze_input(b,bad,163840,&owner);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;table[0]++;try{freeze_input(b,lengths,163840,&owner);}catch(...){table[0]--;throw;}table[0]--;});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;positions[0]++;try{freeze_input(b,lengths,163840,&owner);}catch(...){positions[0]--;throw;}positions[0]--;});
  }
 }
 if(active_spec_proof_r3632||positive!=56||negative!=10)throw std::runtime_error("3632 CPU test counts/TLS cleanup");
 serialized=nlohmann::json{{"completed",true},{"valid_metadata_cases",positive},{"invalid_metadata_cases",negative},{"batch_sizes",{1,2,4,8,16,32,64}},{"widths",{1,2,3,4,5,6,7,8}},{"GPU_math_tested",false}}.dump();return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_spec_contract_report_r3632(){return serialized.c_str();}
