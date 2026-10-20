#include "kimi_k3_trace.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <cmath>
#include <cstdint>
#include <vector>
#include <stdexcept>
namespace infinilm::models::kimi_k3 {
namespace {
size_t option(const char*name,size_t value){auto s=std::getenv(name);return s?std::stoull(s):value;}
}
bool trace_detail_layer(size_t layer){
 // Only the explicitly traced epoch changes detail selection. Preserve the
 // parent's disabled-trace path and its existing range behavior otherwise.
 const char*directory=std::getenv("INFINILM_KIMI_TRACE_DIR");
 const char*cursor=directory&&*directory?std::getenv("INFINILM_KIMI_TRACE_LAYERS"):nullptr;
 if(cursor&&*cursor){
  while(*cursor){char*end=nullptr;auto value=std::strtoull(cursor,&end,10);
   if(end==cursor||(*end&&*end!=','))throw std::invalid_argument("Invalid Kimi trace layer list");
   if(value==layer)return true;cursor=*end?end+1:end;
  }return false;
 }
 return layer>=option("INFINILM_KIMI_TRACE_FIRST",45)&&layer<=option("INFINILM_KIMI_TRACE_LAST",52);
}
void trace_hidden(runtime::Runtime&r,const runtime::Tensor&x,const global_state::ForwardContext&fc,size_t layer,const char*point){
 const char*directory=std::getenv("INFINILM_KIMI_TRACE_DIR");if(!directory||!*directory)return;
 r.assertOwnerThread();if(r.isGraphRecording()||!x||x->dtype()!=runtime::DataType::kBFloat16||x->ndim()<2||!fc.attn_metadata.input_offsets)throw std::logic_error("Kimi trace requires eager BF16 hidden with request offsets");
 size_t hidden=x->size(x->ndim()-1),tokens=x->numel()/hidden;auto contiguous=x->contiguous(r);std::vector<uint16_t>raw(x->numel());r.memcpyD2H(raw.data(),contiguous->data(),raw.size()*2);
 auto offsets=fc.attn_metadata.input_offsets.value()->contiguous(r);if(offsets->dtype()!=runtime::DataType::kInt32)throw std::invalid_argument("Kimi trace offset type differs");std::vector<int32_t>host_offsets(offsets->numel());r.memcpyD2H(host_offsets.data(),offsets->data(),host_offsets.size()*4);
 if(host_offsets.size()<2||host_offsets.front()!=0||host_offsets.back()!=int64_t(tokens))throw std::invalid_argument("Kimi trace packed token count differs");
 auto stage_env=std::getenv("SLURM_PROCID");int stage=stage_env?std::atoi(stage_env):0,rank=r.device().index();
 std::string prefix=std::string(directory)+"/stage"+std::to_string(stage)+"-rank"+std::to_string(rank)+"-layer"+std::to_string(layer)+"-"+point;
 std::vector<uint16_t>last;std::vector<nlohmann::json>requests;
 for(size_t request=0;request+1<host_offsets.size();++request){int32_t begin=host_offsets[request],end=host_offsets[request+1];if(begin<0||begin>=end||size_t(end)>tokens)throw std::invalid_argument("Kimi trace offsets invalid");uint64_t hash=14695981039346656037ULL;size_t nonfinite=0;double sum=0,squares=0;
  for(size_t i=size_t(begin)*hidden;i<size_t(end)*hidden;++i){uint16_t b=raw[i];hash=(hash^uint8_t(b))*1099511628211ULL;hash=(hash^uint8_t(b>>8))*1099511628211ULL;uint32_t bits=uint32_t(b)<<16;float value;std::memcpy(&value,&bits,4);if(!std::isfinite(value))++nonfinite;else{sum+=value;squares+=double(value)*value;}}
  requests.push_back({{"request",request},{"tokens",end-begin},{"fnv1a64",std::to_string(hash)},{"sum",sum},{"sum_squares",squares},{"nonfinite",nonfinite}});last.insert(last.end(),raw.begin()+size_t(end-1)*hidden,raw.begin()+size_t(end)*hidden);
 }
 auto write=[&](const std::string&path,const uint16_t*p,size_t n){std::ofstream f(path,std::ios::binary);f.write(reinterpret_cast<const char*>(p),n*2);if(!f)throw std::runtime_error("Kimi trace write failed: "+path);};
 write(prefix+"-last-bf16.bin",last.data(),last.size());
 // Full snapshots of the first two layers help inspect an earliest mismatch.
 if((rank==0||rank==6)&&((std::getenv("INFINILM_KIMI_TRACE_MLA_FULL")&&std::string(point).rfind("mla_",0)==0)||(layer>=option("INFINILM_KIMI_TRACE_FULL_FIRST",45)&&layer<=option("INFINILM_KIMI_TRACE_FULL_LAST",48)))&&std::string(point).find("residual_slot_")!=0)write(prefix+"-full-bf16.bin",raw.data(),raw.size());
 nlohmann::json record={{"stage",stage},{"rank",rank},{"layer",layer},{"point",point},{"tokens",tokens},{"hidden",hidden},{"requests",requests}};
 std::ofstream f(std::string(directory)+"/stage"+std::to_string(stage)+"-rank"+std::to_string(rank)+".jsonl",std::ios::app);f<<record.dump()<<'\n';if(!f)throw std::runtime_error("Kimi trace journal write failed");
}
}
