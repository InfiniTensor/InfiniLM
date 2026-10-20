#pragma once
#include "runtime/runtime.hpp"
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
extern "C" const char* modern_kimi_reduce_error_r3226();
extern "C" int modern_kimi_reduce_policy_r3226(int);
extern "C" int modern_kimi_reduce_stats_r3226(int,std::uint64_t*,std::size_t);
extern "C" int modern_kimi_reduce_begin_r3226(void*,std::size_t);
extern "C" int modern_kimi_reduce_end_r3226(void*);
namespace infinilm::engine::distributed::reduce_precision_r3226 {
bool enabled(infinilm::runtime::Runtime&);
int selected(infinilm::runtime::Runtime&);
void record_wide(infinilm::runtime::Runtime&,bool,std::size_t);
void record(infinilm::runtime::Runtime&,std::size_t);
class Scope final {
 infinilm::runtime::Runtime* owner_;
public:
 Scope(infinilm::runtime::Runtime&r,std::size_t requests):owner_(&r){
  if(modern_kimi_reduce_begin_r3226(owner_,requests))throw std::runtime_error(modern_kimi_reduce_error_r3226());
 }
 ~Scope()noexcept{if(modern_kimi_reduce_end_r3226(owner_))std::terminate();}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
}

extern "C" int modern_kimi_reduce_wide_stats_r3393(int,std::uint64_t*,std::size_t);
