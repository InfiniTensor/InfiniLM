#include "kimi_k3_kda_trace.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>
#include <stdexcept>
namespace infinilm::models::kimi_k3 {
namespace rt=infinilm::runtime;
struct KdaTraceImpl{
 struct Snapshot{rt::Tensor value;std::string point;};
 rt::Runtime&r;std::string directory;size_t layer,call;int stage,rank;bool finished=false;
 std::vector<Snapshot>snapshots;
 KdaTraceImpl(rt::Runtime&owner,std::string d,size_t l,size_t c):r(owner),directory(std::move(d)),layer(l),call(c),stage(std::getenv("SLURM_PROCID")?std::atoi(std::getenv("SLURM_PROCID")):0),rank(owner.device().index()){}
 void capture(const rt::Tensor&x,const char*point){
  r.assertOwnerThread();if(!x||x->device()!=r.device()||r.isGraphRecording()||snapshots.size()>=64)throw std::logic_error("KDA diagnostic snapshot owner/state differs");
  if(x->dtype()!=rt::DataType::kBFloat16&&x->dtype()!=rt::DataType::kFloat32&&x->dtype()!=rt::DataType::kInt32)throw std::invalid_argument("KDA diagnostic supports BF16, F32 and I32 without conversion");
  auto copy=rt::Tensor::empty(r,x->shape(),x->dtype());copy->copy_from(r,x);snapshots.push_back({copy,point});
  const char*sync=std::getenv("INFINILM_KIMI_KDA_TRACE_SYNC_EACH");if(sync&&std::string(sync)=="1")r.syncStream();
 }
 void finish(){
  r.syncStream();
  std::string stem="stage"+std::to_string(stage)+"-rank"+std::to_string(rank)+"-layer"+std::to_string(layer)+"-kda-call"+std::to_string(call);
  std::ofstream journal(directory+"/"+stem+".jsonl",std::ios::app);
  if(!journal)throw std::runtime_error("Cannot open KDA snapshot journal");
  for(auto&snapshot:snapshots){
   auto&x=snapshot.value;bool bf16=x->dtype()==rt::DataType::kBFloat16,f32=x->dtype()==rt::DataType::kFloat32;size_t itemsize=bf16?2:4;
   std::vector<unsigned char>raw(x->numel()*itemsize);r.memcpyD2H(raw.data(),x->data(),raw.size());uint64_t hash=14695981039346656037ULL;for(auto b:raw)hash=(hash^b)*1099511628211ULL;
   size_t nonfinite=0;for(size_t i=0;i<x->numel()&&(bf16||f32);++i){if(bf16){uint16_t bits;std::memcpy(&bits,raw.data()+i*2,2);if((bits&0x7f80)==0x7f80)++nonfinite;}else{uint32_t bits;std::memcpy(&bits,raw.data()+i*4,4);if((bits&0x7f800000)==0x7f800000)++nonfinite;}}
   auto name=stem+"-"+snapshot.point+"-raw.bin";std::ofstream binary(directory+"/"+name,std::ios::binary);binary.write(reinterpret_cast<const char*>(raw.data()),raw.size());if(!binary)throw std::runtime_error("Cannot write KDA snapshot");
   nlohmann::json row={{"stage",stage},{"rank",rank},{"layer",layer},{"call",call},{"point",snapshot.point},{"shape",x->shape()},{"dtype",bf16?"bf16":f32?"f32":"i32"},{"bytes",raw.size()},{"fnv1a64",std::to_string(hash)},{"nonfinite",nonfinite},{"file",name}};journal<<row.dump()<<'\n';
  }
  if(!journal)throw std::runtime_error("KDA snapshot journal write failed");finished=true;
 }
};
namespace {thread_local KdaTraceImpl*active=nullptr;thread_local std::map<std::string,size_t>calls;}
KdaTraceBatch::KdaTraceBatch(rt::Runtime&r,const global_state::ForwardContext&,size_t layer):previous_(active){
 const char*d=std::getenv("INFINILM_KIMI_TRACE_DIR");const char*s=std::getenv("INFINILM_KIMI_KDA_TRACE_LAYER");size_t selected=s?std::stoull(s):46;
 if(d&&*d&&layer==selected){if(r.isGraphRecording())throw std::logic_error("KDA raw trace requires eager Runtime");std::string key=std::string(d)+"/"+std::to_string(layer);impl_=std::make_unique<KdaTraceImpl>(r,d,layer,calls[key]++);active=impl_.get();}
}
KdaTraceBatch::~KdaTraceBatch(){if(impl_&&!impl_->finished)try{impl_->r.syncStream();}catch(...){std::terminate();}active=previous_;}
void KdaTraceBatch::capture(const rt::Tensor&x,const char*p){if(impl_)impl_->capture(x,p);}
void KdaTraceBatch::finish(){if(impl_)impl_->finish();}
void trace_kda_projection(rt::Runtime&r,const rt::Tensor&x,const char*p){if(active){if(&active->r!=&r)throw std::logic_error("KDA projection trace crossed Runtime owners");active->capture(x,p);}}
}
