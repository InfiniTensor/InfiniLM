#pragma once
#include "runtime/runtime.hpp"
#include <chrono>
#include <type_traits>
#include <utility>
namespace infinilm::models::kimi_k3 {
bool profile_enabled_r290();
void profile_begin_r290(runtime::Runtime&);
void profile_record_r290(runtime::Runtime&,const char*,double);
template<class F> decltype(auto) profile_call_r290(runtime::Runtime&r,const char*label,F&&f) {
 if(!profile_enabled_r290()) return std::forward<F>(f)();
 r.assertOwnerThread();r.syncStream();auto start=std::chrono::steady_clock::now();
 if constexpr(std::is_void_v<std::invoke_result_t<F>>) {
  std::forward<F>(f)();r.syncStream();
  profile_record_r290(r,label,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
 } else {
  auto result=std::forward<F>(f)();r.syncStream();
  profile_record_r290(r,label,std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());
  return result;
 }
}
}
