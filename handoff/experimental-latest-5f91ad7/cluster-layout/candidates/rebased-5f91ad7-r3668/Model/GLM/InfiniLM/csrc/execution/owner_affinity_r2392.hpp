#pragma once
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <pthread.h>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
namespace infinilm::execution::affinity_r2392 {
inline std::array<std::atomic<unsigned>,8> changes{};
inline std::array<std::string,8> masks{};
inline std::mutex mutex;
inline void apply_once(int device){
 static thread_local const int width=[](){const char*p=std::getenv("INFINILM_OWNER_CPU_WIDTH_R2392");if(!p)return 0;const std::string s=p;if(s=="0")return 0;if(s=="1")return 1;if(s=="4")return 4;if(s=="8")return 8;throw std::invalid_argument("2392 CPU width must be explicitly 0,1,4,8");}();
 if(!width)return;
 static thread_local int applied_device=-1;
 if(applied_device!=-1){if(applied_device!=device)throw std::logic_error("2392 submitting thread changed Runtime device");return;}
 if(device<0||device>=8)throw std::invalid_argument("2392 only eight-device model Runtime owners allowed");
 cpu_set_t original{},desired{},observed{};if(pthread_getaffinity_np(pthread_self(),sizeof(original),&original))throw std::runtime_error("2392 cannot query owner affinity");
 std::vector<int> allowed;for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&original))allowed.push_back(cpu);
 if(allowed.size()<static_cast<size_t>(8*width))throw std::invalid_argument("2392 owner mask too narrow for eight disjoint requested subsets");
 for(int i=0;i<width;++i)CPU_SET(allowed.at(device*width+i),&desired);
 if(pthread_setaffinity_np(pthread_self(),sizeof(desired),&desired)||pthread_getaffinity_np(pthread_self(),sizeof(observed),&observed)||!CPU_EQUAL(&observed,&desired))throw std::runtime_error("2392 owner affinity set/readback mismatch");
 std::ostringstream s;bool first=true;for(int cpu=0;cpu<CPU_SETSIZE;++cpu)if(CPU_ISSET(cpu,&observed)){if(!first)s<<',';first=false;s<<cpu;}
 {std::lock_guard<std::mutex> lock(mutex);masks.at(device)=s.str();changes.at(device).fetch_add(1);}
 applied_device=device;
}
}
