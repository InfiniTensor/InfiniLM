// Included once by the isolated Common communication_group.cpp. All model
// DSOs call these external APIs, so the request frame cannot split by DSO.
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
namespace infinilm::engine::distributed::reduce_precision_r3226 {
namespace {
using RT=infinilm::runtime::Runtime;
std::mutex policy_mutex;
std::atomic<int> policy{0};
std::atomic<std::uint64_t> switches{0};
std::size_t in_flight{};
struct Counters{std::atomic<std::uint64_t>calls{0},bytes{0},peak{0};};
std::array<Counters,8>counters;
struct Frame{RT*owner{};bool promote{};};
thread_local Frame frame;
thread_local std::string error;
}
bool enabled(RT&r){
 r.assertOwnerThread();if(!frame.owner)return false;
 if(frame.owner!=&r)throw std::logic_error("3226 different Runtime inside request frame");
 if(frame.promote&&r.graphCaptureState()!=infinilm::execution::graph::GraphManager::CaptureState::kInactive)
  throw std::logic_error("3226 FP32 reduction requires eager owner");
 return frame.promote;
}
void record(RT&r,std::size_t bytes){
 r.assertOwnerThread();if(!enabled(r))throw std::logic_error("3226 record outside promoted frame");
 auto&c=counters.at(r.device().index());++c.calls;c.bytes.fetch_add(bytes);
 auto peak=c.peak.load();while(peak<bytes&&!c.peak.compare_exchange_weak(peak,bytes)){}
}
int set(int flag){
 if(flag!=0&&flag!=1){error="3226 invalid policy";return 1;}
 std::lock_guard<std::mutex>lock(policy_mutex);
 if(in_flight){error="3226 policy during active forward";return 2;}
 policy.store(flag);++switches;return 0;
}
int begin(void*raw,std::size_t requests){try{
 if(!raw||!requests||requests>128)throw std::invalid_argument("3226 invalid request frame");
 auto&r=*static_cast<RT*>(raw);r.assertOwnerThread();
 if(frame.owner)throw std::logic_error("3226 nested request frame");
 std::lock_guard<std::mutex>lock(policy_mutex);
 const bool promote=requests>1&&policy.load()==1;
 if(promote&&(r.device().type()!=infinilm::runtime::Device::Type::kHygon||r.device().index()<0||r.device().index()>=8||r.graphCaptureState()!=infinilm::execution::graph::GraphManager::CaptureState::kInactive))
  throw std::logic_error("3226 FP32 multi-request requires eager Hygon owner");
 frame={&r,promote};++in_flight;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
int end(void*raw){try{
 if(!raw||frame.owner!=raw)throw std::invalid_argument("3226 mismatched request frame end");
 frame.owner->assertOwnerThread();std::lock_guard<std::mutex>lock(policy_mutex);
 if(!in_flight)throw std::logic_error("3226 frame underflow");
 frame={};--in_flight;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
int stats(int device,std::uint64_t*out,std::size_t count){try{
 if(device<0||device>=8||!out||count!=5)throw std::invalid_argument("3226 invalid stats bounds");
 auto&c=counters[device];std::array<std::uint64_t,5>v{std::uint64_t(policy.load()),c.calls.load(),c.bytes.load(),c.peak.load(),switches.load()};std::copy(v.begin(),v.end(),out);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
const char*message(){return error.c_str();}
}
extern "C" const char*modern_kimi_reduce_error_r3226(){return infinilm::engine::distributed::reduce_precision_r3226::message();}
extern "C" int modern_kimi_reduce_policy_r3226(int flag){return infinilm::engine::distributed::reduce_precision_r3226::set(flag);}
extern "C" int modern_kimi_reduce_begin_r3226(void*r,std::size_t requests){return infinilm::engine::distributed::reduce_precision_r3226::begin(r,requests);}
extern "C" int modern_kimi_reduce_end_r3226(void*r){return infinilm::engine::distributed::reduce_precision_r3226::end(r);}
extern "C" int modern_kimi_reduce_stats_r3226(int d,std::uint64_t*out,std::size_t n){return infinilm::engine::distributed::reduce_precision_r3226::stats(d,out,n);}
