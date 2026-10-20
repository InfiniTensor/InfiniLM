#pragma once
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include <array>
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <typeinfo>
namespace infinilm::diagnostics::host_r1551 {
using Clock=std::chrono::steady_clock;using json=nlohmann::json;
inline std::atomic<bool> enabled{false};
struct Entry {uint64_t calls=0,ns=0;};
struct Frame {runtime::Runtime*owner;std::thread::id thread;Clock::time_point begin;std::map<std::string,Entry> entries;uint64_t serial;};
inline thread_local Frame*current=nullptr;inline thread_local uint64_t sequence=0;
inline std::array<std::mutex,8>locks;inline std::array<json,8>reports;
struct Scope {
 runtime::Runtime&r;Frame frame;bool active;
 explicit Scope(runtime::Runtime&v):r(v),frame{&v,std::this_thread::get_id(),Clock::now(),{},++sequence},active(enabled.load(std::memory_order_relaxed)) {
  r.assertOwnerThread();if(current)throw std::logic_error("1551 nested owner profile scope");
  if(active){if(r.isGraphRecording())throw std::logic_error("1551 profile forbids graph recording");current=&frame;}
 }
 ~Scope()noexcept{
  if(!active)return;
  if(current!=&frame||std::this_thread::get_id()!=frame.thread)std::terminate();
  const auto total=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-frame.begin).count();current=nullptr;
  try{json rows=json::object();for(auto&[name,e]:frame.entries)rows[name]={{"calls",e.calls},{"host_nanoseconds",e.ns}};
   json report={{"device",r.device().index()},{"serial",frame.serial},{"owner_scope_host_nanoseconds",total},{"entries",rows},{"GPU_events_or_added_drains",0},{"scope","Host wall durations, overlapping nested scopes; asynchronous submission time is not GPU execution time. Existing explicit sync calls may include queued device and peer wait. Shared-library/internal sync coverage not claimed."}};
   std::lock_guard<std::mutex>lock(locks.at(r.device().index()));reports.at(r.device().index())=std::move(report);
  }catch(...){std::terminate();}
 }
};
struct Timer {
 Frame*frame;const char*name;Clock::time_point begin;
 explicit Timer(const char*value):frame(current),name(value){if(frame)begin=Clock::now();}
 ~Timer()noexcept{if(frame){if(current!=frame||std::this_thread::get_id()!=frame->thread)std::terminate();try{auto&e=frame->entries[name];++e.calls;e.ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-begin).count();}catch(...){std::terminate();}}}
};
template<class Call>auto call(const char*name,Call&&fn){Timer timer{name};return fn();}
inline void sync(runtime::Runtime&r){if(current&&current->owner!=&r)throw std::logic_error("1551 foreign Runtime sync");Timer timer{"explicit_Runtime_syncStream"};r.syncStream();}
inline json latest(int device){std::lock_guard<std::mutex>lock(locks.at(device));return reports.at(device);}
}
