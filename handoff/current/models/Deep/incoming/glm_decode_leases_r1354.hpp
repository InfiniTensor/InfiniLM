#pragma once
#include "runtime/runtime.hpp"
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

// Private opt-in decode experiment. The caller keeps all borrowed inputs alive
// until finish(). A successful Work transfers its owned temporaries here;
// exceptions always drain before any Work temporaries can be released.
namespace infinilm::models::glm_moe_dsa::decode_leases_r1354 {
namespace rt=infinilm::runtime;
struct Stats { size_t retired_groups=0,retired_tensors=0,logical_bytes=0,peak_logical_bytes=0,budget_drains=0,success_fences_deferred=0,final_drains=0,exception_drains=0; };
struct Frame { rt::Runtime*owner=nullptr;void*stream=nullptr;size_t limit=0,bytes=0;std::vector<rt::Tensor>held;Stats stats; };
inline thread_local Frame frame;
inline thread_local Stats last;
inline bool active(rt::Runtime&r) {
 r.assertOwnerThread();if(!frame.owner)return false;
 if(frame.owner!=&r||frame.stream!=r.stream())throw std::logic_error("1354 foreign Runtime/stream");
 if(r.isGraphRecording())throw std::logic_error("1354 forbids graph recording");return true;
}
inline void drain_release(rt::Runtime&r) {
 if(!active(r))throw std::logic_error("1354 no active frame");r.syncStream();frame.held.clear();frame.bytes=0;
}
inline void begin(rt::Runtime&r,size_t limit) {
 r.assertOwnerThread();if(frame.owner||!frame.held.empty())throw std::logic_error("1354 nested/live frame");
 if(!limit||r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon)throw std::invalid_argument("1354 invalid owner/budget");
 frame.stats={};frame.owner=&r;frame.stream=r.stream();frame.limit=limit;frame.bytes=0;
}
inline void success_fence(rt::Runtime&r) {if(active(r))++frame.stats.success_fences_deferred;else r.syncStream();}
inline void retire(rt::Runtime&r,std::vector<rt::Tensor>&held,int entered_exceptions)noexcept {
 try {
  if(!active(r)||std::uncaught_exceptions()>entered_exceptions){r.syncStream();if(frame.owner==&r)++frame.stats.exception_drains;return;}
  size_t bytes=0;for(auto&t:held)if(t){if(t->device()!=r.device())throw std::logic_error("1354 foreign Tensor device");if(t->nbytes()>std::numeric_limits<size_t>::max()-bytes)throw std::overflow_error("1354 byte overflow");bytes+=t->nbytes();}
  if(bytes>frame.limit||frame.bytes>frame.limit-bytes){drain_release(r);++frame.stats.budget_drains;}
  // A group larger than the cap is kept local and drained, never retained.
  if(bytes>frame.limit){r.syncStream();return;}
  try {frame.held.reserve(frame.held.size()+held.size());}catch(...){r.syncStream();return;}
  ++frame.stats.retired_groups;frame.stats.retired_tensors+=held.size();frame.stats.logical_bytes+=bytes;frame.bytes+=bytes;
  if(frame.bytes>frame.stats.peak_logical_bytes)frame.stats.peak_logical_bytes=frame.bytes;
  for(auto&t:held)frame.held.push_back(std::move(t));held.clear();
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}std::terminate();}
}
inline Stats finish(rt::Runtime&r) {
 if(!active(r))throw std::logic_error("1354 finish without owner");drain_release(r);++frame.stats.final_drains;last=frame.stats;frame.owner=nullptr;frame.stream=nullptr;frame.limit=0;return last;
}
inline void abort(rt::Runtime&r)noexcept {if(frame.owner!=&r)return;try{r.assertOwnerThread();drain_release(r);++frame.stats.exception_drains;last=frame.stats;frame.owner=nullptr;frame.stream=nullptr;frame.limit=0;}catch(...){std::terminate();}}
struct Guard {
 rt::Runtime&r;bool enabled;
 Guard(rt::Runtime&owner,bool use,size_t limit=256*1024*1024):r(owner),enabled(use){r.assertOwnerThread();if(frame.owner)throw std::logic_error("1354 overlapping frame");if(enabled)begin(r,limit);}
 ~Guard(){if(enabled)abort(r);}
 Stats finish(){if(!enabled)return {};auto out=decode_leases_r1354::finish(r);enabled=false;return out;}
};
struct Work {
 rt::Runtime&r;std::vector<rt::Tensor>held;int entered_exceptions=std::uncaught_exceptions();
 ~Work(){retire(r,held,entered_exceptions);}
 rt::Tensor alloc(rt::Shape shape,rt::DataType type){auto value=rt::Tensor::empty(r,shape,type);held.push_back(value);return value;}
};
}
