#pragma once
#include "runtime/runtime.hpp"
#include "engine/distributed/communication_group.hpp"
#include "execution/submit.hpp"
#include <array>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <stdexcept>
namespace infinilm::models::deepseek_v4::wide_r3443 {
namespace rt=infinilm::runtime;
inline std::atomic<int> policy{0};
inline std::mutex policy_mutex;
inline size_t in_flight{};
struct Counts{std::array<std::atomic<uint64_t>,6>v{};};
inline std::array<Counts,8> counts;
struct Frame{rt::Runtime*owner;int mode;};
inline thread_local const Frame*active=nullptr;
class Scope final {
 Frame f_;
 public:
 Scope(rt::Runtime&r,size_t requests):f_{&r,0}{
  r.assertOwnerThread();
  if(!requests||active||r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||r.device().index()<0||r.device().index()>=8)throw std::invalid_argument("3443 finite eager Hygon owner frame");
  std::lock_guard<std::mutex>lock(policy_mutex);f_.mode=requests>1?policy.load():0;++in_flight;active=&f_;
 }
 ~Scope()noexcept{if(active!=&f_)std::terminate();active=nullptr;std::lock_guard<std::mutex>lock(policy_mutex);if(!in_flight)std::terminate();--in_flight;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
inline int selected(rt::Runtime&r){r.assertOwnerThread();if(!active)return 0;if(active->owner!=&r)throw std::logic_error("3443 different Runtime in request frame");return active->mode;}
inline int set(int value){if(value<0||value>4)return 1;std::lock_guard<std::mutex>lock(policy_mutex);if(in_flight)return 2;policy.store(value);return 0;}
// 0 original; 1 matched synchronization with native arithmetic;
// 2 attention F64; 3 attention+FFN F64; 4 attention+FFN+indexer F64.
// All modes 1..4 drain each instrumented reduction, keeping a scheduling control.
inline void reduce(rt::Runtime&r,const rt::Tensor&value,infinicclComm_t comm,int kind){
 const int mode=selected(r);
 if(!mode){infinilm::engine::distributed::allreduce_(r,value,value,infinicclSum,comm);return;}
 if(kind<0||kind>2||!value||(value->dtype()!=rt::DataType::kBFloat16&&value->dtype()!=rt::DataType::kFloat32))throw std::invalid_argument("3443 bounded SUM source dtype/kind");
 const bool wide=mode>=kind+2;rt::Tensor scratch;
 try{
  if(wide){
   scratch=rt::Tensor::empty(r,value->shape(),rt::DataType::kFloat64);
   auto cfg=infini::ops::Cast::DefaultConfig(r.device().type());
   infinilm::execution::submit<infini::ops::Cast>(r,cfg,value,scratch);
   infinilm::engine::distributed::allreduce_(r,scratch,scratch,infinicclSum,comm);
   infinilm::execution::submit<infini::ops::Cast>(r,cfg,scratch,value);
  }else infinilm::engine::distributed::allreduce_(r,value,value,infinicclSum,comm);
  r.syncStream();auto&v=counts.at(r.device().index()).v;++v[0];
  if(wide){++v[1+kind];const auto bytes=scratch->nbytes();v[4].fetch_add(bytes);auto peak=v[5].load();while(peak<bytes&&!v[5].compare_exchange_weak(peak,bytes)){};}
 }catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
inline int stats(int device,uint64_t*out,size_t size){
 if(device<0||device>=8||!out||size!=6)return 1;
 std::array<uint64_t,6> result{};for(size_t i=0;i<6;++i)result[i]=counts[device].v[i].load();std::copy(result.begin(),result.end(),out);return 0;
}
}
