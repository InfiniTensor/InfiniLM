#pragma once
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace infinilm::models::kimi_k3 {
namespace {
inline thread_local std::map<const void*,size_t> route_layers_r585;
inline std::array<std::atomic<uint64_t>,8> route_records_r585{};
struct RouteProfileR585 {
 runtime::Runtime&r;bool active=false;std::string directory,tag;size_t layer=0;
 std::chrono::steady_clock::time_point last;nlohmann::json row;
 RouteProfileR585(runtime::Runtime&owner,const void*object,size_t tokens):r(owner){
  const char*root=std::getenv("INFINILM_KIMI_ROUTE_DIR_R585");
  const char*label=std::getenv("INFINILM_KIMI_ROUTE_TAG_R585");
  if(!root||!*root||!label||!*label||!tokens||tokens>64)return;
  r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("Route profile requires eager owner");
  directory=root;tag=label;layer=route_layers_r585.at(object);active=true;
  const char*stage=std::getenv("SLURM_PROCID");
  row={{"tag",tag},{"layer",layer},{"rank",r.device().index()},{"stage",stage?std::atoi(stage):0},{"tokens",tokens},{"sections_ms",nlohmann::json::object()}};
  r.syncStream();last=std::chrono::steady_clock::now();
 }
 void mark(const char*name){if(!active)return;r.syncStream();auto now=std::chrono::steady_clock::now();row["sections_ms"][name]=std::chrono::duration<double,std::milli>(now-last).count();last=now;}
 void capture(const runtime::Tensor&ids,const runtime::Tensor&weights,size_t experts,size_t hidden,size_t intermediate,size_t parameter_bytes){
  if(!active)return;
  if(ids->dtype()!=runtime::DataType::kInt32||weights->dtype()!=runtime::DataType::kFloat32||ids->ndim()!=2||weights->shape()!=ids->shape()||!ids->is_contiguous()||!weights->is_contiguous()||!experts)throw std::invalid_argument("Route profile metadata differs");
  std::vector<int32_t>selected(ids->numel());std::vector<float>routing(weights->numel());
  r.memcpyD2H(selected.data(),ids->data(),selected.size()*4);r.memcpyD2H(routing.data(),weights->data(),routing.size()*4);
  std::vector<size_t>counts(experts,0);size_t topk=ids->size(1);
  for(size_t t=0;t<ids->size(0);++t){std::set<int32_t>seen;for(size_t k=0;k<topk;++k){size_t index=t*topk+k;int32_t e=selected[index];if(e<0||size_t(e)>=experts||!seen.insert(e).second||!std::isfinite(routing[index]))throw std::invalid_argument("Route profile indices/weights invalid");++counts[size_t(e)];}}
  auto hash=[](const void*p,size_t n){uint64_t h=14695981039346656037ULL;auto b=static_cast<const unsigned char*>(p);for(size_t i=0;i<n;++i)h=(h^b[i])*1099511628211ULL;return std::to_string(h);};
  nlohmann::json histogram=nlohmann::json::object();size_t unique=0;for(auto c:counts){if(c)++unique;auto key=std::to_string(c);histogram[key]=histogram.value(key,size_t{0})+1;}
  row["experts"]=experts;row["hidden"]=hidden;row["local_intermediate"]=intermediate;row["topk"]=topk;row["unique_experts"]=unique;row["expert_count_histogram"]=histogram;
  row["selected_experts"]=selected;row["routing_weights"]=routing;row["IDs_fnv1a64"]=hash(selected.data(),selected.size()*4);row["weights_fnv1a64"]=hash(routing.data(),routing.size()*4);
  row["local_all_expert_parameter_bytes"]=parameter_bytes;
  row["unique_expert_parameter_bytes"]=double(parameter_bytes)/experts*unique;
  row["per_route_parameter_bytes"]=double(parameter_bytes)/experts*selected.size();
  mark("route_observation_ms");
 }
 void finish(){if(!active)return;mark("final_add_ms");double total=0;for(auto&v:row["sections_ms"].items())total+=v.value().get<double>();row["total_observed_ms"]=total;
  std::ofstream f(directory+"/stage"+std::to_string(row["stage"].get<int>())+"-rank"+std::to_string(r.device().index())+".jsonl",std::ios::app);f<<row.dump()<<'\n';if(!f)throw std::runtime_error("Route profile write failed");route_records_r585.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);
 }
};
}
}
