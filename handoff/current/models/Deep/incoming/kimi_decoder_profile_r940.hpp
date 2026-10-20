#pragma once
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
namespace infinilm::models::kimi_k3 {
struct DecoderProfileR940 {
 runtime::Runtime&r;bool active=false;std::string directory;nlohmann::json row;std::chrono::steady_clock::time_point last;
 DecoderProfileR940(runtime::Runtime&owner,size_t layer,size_t tokens,bool kda,bool moe):r(owner){
  auto root=std::getenv("INFINILM_KIMI_ROUTE_DIR_R585"),tag=std::getenv("INFINILM_KIMI_ROUTE_TAG_R585");if(!root||!*root||!tag||!*tag||!tokens||tokens>64)return;
  r.assertOwnerThread();if(r.isGraphRecording())throw std::logic_error("940profile eager only");active=true;directory=root;auto stage=std::getenv("SLURM_PROCID");row={{"tag",tag},{"layer",layer},{"rank",r.device().index()},{"stage",stage?std::atoi(stage):0},{"tokens",tokens},{"is_kda",kda},{"is_moe",moe},{"sections_ms",nlohmann::json::object()}};r.syncStream();last=std::chrono::steady_clock::now();
 }
 void mark(const char*name){if(!active)return;r.syncStream();auto now=std::chrono::steady_clock::now();row["sections_ms"][name]=std::chrono::duration<double,std::milli>(now-last).count();last=now;}
 void finish(){if(!active)return;mark("final_add_ms");double total=0;for(auto&v:row["sections_ms"].items())total+=v.value().get<double>();row["total_observed_ms"]=total;std::ofstream f(directory+"/decoder-stage"+std::to_string(row["stage"].get<int>())+"-rank"+std::to_string(r.device().index())+".jsonl",std::ios::app);f<<row.dump()<<'\n';if(!f)throw std::runtime_error("940profile write failed");}
};
}
