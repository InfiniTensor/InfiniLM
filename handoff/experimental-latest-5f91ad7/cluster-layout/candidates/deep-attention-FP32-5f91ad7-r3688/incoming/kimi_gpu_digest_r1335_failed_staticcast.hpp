#pragma once
#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
extern "C" int modern_kimi_gpu_hash_r1334(void*,size_t,const void*,size_t,size_t,void*);
namespace infinilm::diagnostics::kimi_digest_r1335 {
using json=nlohmann::json;
constexpr size_t slots=96,payload_bytes=slots*16,padding=256;
struct Frame{runtime::Runtime*owner=nullptr;runtime::Tensor arena;int mode=0,tp_rank=0,pp_stage=0;uint64_t serial=0;std::vector<json>points;};
inline thread_local Frame frame;inline thread_local uint64_t sequence=0;inline std::array<std::mutex,8>report_mutex;inline std::array<json,8>reports;
inline void publish(runtime::Runtime&r,json value){std::lock_guard<std::mutex>lock(report_mutex.at(r.device().index()));reports.at(r.device().index())=std::move(value);}
inline json latest(int device){std::lock_guard<std::mutex>lock(report_mutex.at(device));return reports.at(device);}
inline void abort(runtime::Runtime&r)noexcept{
 if(frame.owner!=&r)return;try{r.assertOwnerThread();infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};r.syncStream();frame.arena.reset();frame.points.clear();frame.owner=nullptr;}catch(...){std::terminate();}
}
inline void begin(runtime::Runtime&r,int mode,int tp_rank,int pp_stage){
 r.assertOwnerThread();if(frame.owner||frame.arena)throw std::logic_error("1335 previous frame still live");if(mode<0||mode>2)throw std::invalid_argument("1335 mode");if(r.isGraphRecording())throw std::logic_error("1335 diagnostics forbid graph recording");const auto serial=++sequence;
 if(!mode){publish(r,{{"mode",0},{"serial",serial},{"GPU_hashes_recorded",false},{"arena_bytes",0},{"points",json::array()}});return;}
 frame.owner=&r;frame.mode=mode;frame.tp_rank=tp_rank;frame.pp_stage=pp_stage;frame.serial=serial;
 try{infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};frame.points.clear();frame.points.reserve(slots);frame.arena=runtime::Tensor::empty(r,{payload_bytes+2*padding},runtime::DataType::kUInt8);r.setDeviceMemory(frame.arena->data(),0xA5,frame.arena->nbytes());}catch(...){abort(r);throw;}
}
inline void mark(runtime::Runtime&r,const runtime::Tensor&value,size_t layer,const char*kind){
 r.assertOwnerThread();if(!frame.owner)return;if(frame.owner!=&r)throw std::logic_error("1335 foreign Runtime");if(!value||value->device()!=r.device()||value->dtype()!=runtime::DataType::kBFloat16||value->ndim()!=3||value->size(0)!=1||!value->size(1)||!value->size(2)||value->stride(1)<=0||value->stride(2)<=0||frame.points.size()>=slots)throw std::invalid_argument("1335 BF16 last-token view geometry");
 infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};const auto slot=frame.points.size();frame.points.push_back({{"kind",kind},{"layer",layer},{"shape",value->shape()},{"strides",value->strides()},{"slot",slot}});
 if(frame.mode==2){const auto*row=static_cast<const uint16_t*>(value->data())+(value->size(1)-1)*value->stride(1);auto*out=static_cast<unsigned char*>(frame.arena->data())+padding;if(modern_kimi_gpu_hash_r1334(out,slot,row,value->size(2),value->stride(2),r.stream()))throw std::runtime_error("1335 GPU digest launch failed");}
}
inline json finish(runtime::Runtime&r){
 r.assertOwnerThread();if(!frame.owner)return latest(r.device().index());if(frame.owner!=&r)throw std::logic_error("1335 foreign frame finish");
 try{infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};r.syncStream();std::vector<unsigned char>host(frame.arena->nbytes());r.memcpyD2H(host.data(),frame.arena->data(),host.size());for(size_t i=0;i<padding;++i)if(host[i]!=0xA5||host[host.size()-1-i]!=0xA5)throw std::runtime_error("1335 arena guards");json points=json::array();for(const auto&p:frame.points){auto point=p;std::array<uint64_t,2>words{};std::memcpy(words.data(),host.data()+padding+p.at("slot").get<size_t>()*16,16);point["GPU_hash"]=words;points.push_back(point);}bool untouched=true;for(size_t i=padding;i<padding+payload_bytes;++i)untouched&=host[i]==0xA5;if(frame.mode==1&&!untouched)throw std::runtime_error("1335 reserve-only buffer mutated");
 json value={{"mode",frame.mode},{"serial",frame.serial},{"device",r.device().index()},{"TP_rank",frame.tp_rank},{"PP_stage",frame.pp_stage},{"arena_bytes",host.size()},{"arena_guards_unchanged",true},{"reserve_only_payload_untouched",untouched},{"GPU_hashes_recorded",frame.mode==2},{"points",points},{"frame_end_explicit_drains",1},{"per_point_explicit_drains",0},{"per_point_D2H_copies",0},{"per_point_GPU_allocations",0}};frame.arena.reset();frame.points.clear();frame.owner=nullptr;value["arena_released_before_return"]=true;publish(r,value);return value;
 }catch(...){abort(r);throw;}
}
struct Guard{runtime::Runtime&r;bool done=false;Guard(runtime::Runtime&v,int mode,int tp_rank,int pp_stage):r(v){begin(r,mode,tp_rank,pp_stage);}void finish(){kimi_digest_r1335::finish(r);done=true;}~Guard()noexcept{if(!done)abort(r);}};
}
