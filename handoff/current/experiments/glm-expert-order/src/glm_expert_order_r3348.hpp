#pragma once
#include "runtime/runtime.hpp"
#include <array>
#include <cstdint>
#include <map>
namespace infinilm::models::glm_moe_dsa::expert_order_r3348 {
struct State {bool enabled=false;uint64_t sorts=0,gates=0,downs=0,routes=0,peak=0;};
inline thread_local std::map<runtime::Runtime*,State>states;
inline bool enabled(runtime::Runtime&r,size_t tokens){r.assertOwnerThread();auto i=states.find(&r);return tokens>=512&&tokens<=4096&&i!=states.end()&&i->second.enabled;}
inline void policy(runtime::Runtime&r,bool value){r.assertOwnerThread();states[&r].enabled=value;}
inline void sorted(runtime::Runtime&r,size_t routes){auto&s=states.at(&r);++s.sorts;s.routes+=routes;s.peak=std::max(s.peak,uint64_t(routes*8));}
inline void gate(runtime::Runtime&r){++states.at(&r).gates;}
inline void down(runtime::Runtime&r){++states.at(&r).downs;}
inline std::array<uint64_t,6>stats(runtime::Runtime&r){r.assertOwnerThread();auto i=states.find(&r);if(i==states.end())return{};auto&s=i->second;return{uint64_t(s.enabled),s.sorts,s.gates,s.downs,s.routes,s.peak};}
inline void release(runtime::Runtime&r){r.assertOwnerThread();states.erase(&r);}
}
