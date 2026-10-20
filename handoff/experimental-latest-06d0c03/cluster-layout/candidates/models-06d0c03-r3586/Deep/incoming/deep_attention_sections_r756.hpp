#pragma once
#include "runtime/runtime.hpp"
#include <cuda_runtime_api.h>
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>
namespace infinilm::models::deepseek_v4 {
inline thread_local std::map<const void*,size_t> section_layers_r756;
class AttentionSectionsR756 {
 using Clock=std::chrono::steady_clock;
 runtime::Runtime&r_;std::string kind_,tag_;size_t layer_,tokens_;std::vector<cudaEvent_t>events_;std::vector<std::string>names_;std::vector<double>host_;Clock::time_point last_;bool active_{false};
 void event(){r_.assertOwnerThread();infini::rt::DeviceGuard guard{r_.device().type(),r_.device().index(),true};cudaEvent_t e{};if(cudaEventCreate(&e)!=cudaSuccess)throw std::runtime_error("726 event create failed");events_.push_back(e);if(cudaEventRecord(e,static_cast<cudaStream_t>(r_.stream()))!=cudaSuccess)throw std::runtime_error("726 event record failed");}
 public:
 AttentionSectionsR756(runtime::Runtime&r,const void*object,const char*kind,size_t tokens):r_(r),kind_(kind),layer_(section_layers_r756.at(object)),tokens_(tokens){const char*tag=std::getenv("DEEP_SECTIONS_R756_TAG");if(tag&&*tag){tag_=tag;active_=true;event();last_=Clock::now();}}
 ~AttentionSectionsR756(){infini::rt::DeviceGuard guard{r_.device().type(),r_.device().index(),true};for(auto e:events_)if(cudaEventDestroy(e)!=cudaSuccess)std::terminate();}
 void mark(const char*name){if(!active_)return;auto now=Clock::now();host_.push_back(std::chrono::duration<double>(now-last_).count());names_.push_back(name);event();last_=Clock::now();}
 void finish(){if(!active_)return;r_.assertOwnerThread();infini::rt::DeviceGuard guard{r_.device().type(),r_.device().index(),true};if(cudaEventSynchronize(events_.back())!=cudaSuccess)throw std::runtime_error("726 section drain failed");nlohmann::json rows=nlohmann::json::array();for(size_t i=0;i<names_.size();++i){float ms=0;if(cudaEventElapsedTime(&ms,events_[i],events_[i+1])!=cudaSuccess)throw std::runtime_error("726 elapsed failed");rows.push_back({{"section",names_[i]},{"host_seconds",host_[i]},{"stream_elapsed_ms",ms}});}const char*job=std::getenv("SLURM_JOB_ID"),*step=std::getenv("SLURM_STEP_ID");std::string path="/refactor/results/deep-sections-r756-"+std::string(job)+"."+step+"-rank"+std::to_string(r_.device().index())+".jsonl";std::ofstream out(path,std::ios::app);out<<nlohmann::json{{"tag",tag_},{"kind",kind_},{"layer",layer_},{"tokens",tokens_},{"sections",rows}}.dump()<<'\n';if(!out)throw std::runtime_error("726 journal write failed");}
};
}
