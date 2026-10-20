#pragma once
#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <string>
#include <vector>

namespace infinilm::diagnostics::kimi_event_r1143 {
using Clock=std::chrono::steady_clock;
struct Boundary{std::string name;size_t event;double cpu_ms;};
struct Region{std::string scope;size_t layer,tokens,first;std::vector<Boundary>points;};
struct State{runtime::Runtime*owner=nullptr;std::string tag;std::vector<cudaEvent_t>events;size_t used=0;std::vector<Region>regions;Clock::time_point begin;};
inline thread_local State state;
inline void check(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
inline void abort()noexcept{
 if(state.owner){infini::rt::DeviceGuard guard{state.owner->device().type(),state.owner->device().index(),true};for(auto event:state.events)if(cudaEventDestroy(event)!=cudaSuccess)std::terminate();}else if(!state.events.empty())std::terminate();
 state.events.clear();state.regions.clear();state.owner=nullptr;state.used=0;state.tag.clear();
}
inline size_t record(runtime::Runtime&r){
 r.assertOwnerThread();if(state.owner!=&r||state.used>=state.events.size())throw std::logic_error("1143event owner/capacity");
 infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 const size_t index=state.used++;check(cudaEventRecord(state.events[index],reinterpret_cast<cudaStream_t>(r.stream())));return index;
}
inline void begin(runtime::Runtime&r){
 r.assertOwnerThread();if(state.owner||!state.events.empty())throw std::logic_error("1143previousframe notretired");
 const auto*tag=std::getenv("INFINILM_KIMI_ROUTE_TAG_R585");if(!tag||!*tag)return;
 infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 int device=-1;check(cudaGetDevice(&device));if(device!=r.device().index())throw std::logic_error("1143event currentdevice differs fromowner");
 if(r.isGraphRecording())throw std::logic_error("1143 diagnostic eager only");
 state.owner=&r;state.tag=tag;state.events.reserve(512);
 try{for(size_t i=0;i<512;++i){cudaEvent_t e{};check(cudaEventCreate(&e));state.events.push_back(e);}state.begin=Clock::now();record(r);}
 catch(...){abort();throw;}
}
struct Profile{
 runtime::Runtime&r;bool active=false;Region row;Clock::time_point last;
 Profile(runtime::Runtime&owner,size_t layer,size_t tokens,const char*scope):r(owner){
  if(!state.owner)return;r.assertOwnerThread();if(state.owner!=&r)throw std::logic_error("1143foreignowner region");active=true;row.scope=scope;row.layer=layer;row.tokens=tokens;row.first=record(r);last=Clock::now();
 }
 void mark(const char*name){if(!active)return;size_t event=record(r);auto now=Clock::now();row.points.push_back({name,event,std::chrono::duration<double,std::milli>(now-last).count()});last=now;}
 void finish(){if(!active)return;mark("final_add_ms");state.regions.push_back(std::move(row));active=false;}
};
inline std::string finish(runtime::Runtime&r){
 r.assertOwnerThread();if(!state.owner)return {};if(state.owner!=&r)throw std::logic_error("1143finish owner");
 infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};
 try{
  const size_t end=record(r);check(cudaEventSynchronize(state.events[end])); // One frame-end wait; no per-region waits.
  nlohmann::json out={{"tag",state.tag},{"device",r.device().index()},{"events_used",state.used},{"regions",nlohmann::json::array()},{"per_region_explicit_waits",0},{"per_layer_file_writes",0},{"event_pool_created_before_model",512}};
  float span=0;check(cudaEventElapsedTime(&span,state.events[0],state.events[end]));out["frame_GPU_span_ms"]=span;out["frame_CPU_ms"]=std::chrono::duration<double,std::milli>(Clock::now()-state.begin).count();
  for(const auto&region:state.regions){nlohmann::json item={{"scope",region.scope},{"layer",region.layer},{"tokens",region.tokens},{"sections",nlohmann::json::array()}};size_t previous=region.first;
   for(const auto&point:region.points){float elapsed=0;check(cudaEventElapsedTime(&elapsed,state.events[previous],state.events[point.event]));item["sections"].push_back({{"name",point.name},{"GPU_stream_ms",elapsed},{"CPU_enqueue_ms",point.cpu_ms}});previous=point.event;}
   out["regions"].push_back(std::move(item));
  }
  auto text=out.dump();abort();return text;
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}abort();throw;}
}
}
