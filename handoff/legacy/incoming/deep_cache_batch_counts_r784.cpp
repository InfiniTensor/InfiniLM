#include <array>
#include <atomic>
#include <cstdint>
namespace {std::array<std::array<std::atomic<uint64_t>,8>,3>counts{};}
extern "C" void modern_deep_cache_batch_count_r784(int kind,int device){counts.at(kind).at(device).fetch_add(1,std::memory_order_relaxed);}
extern "C" uint64_t modern_deep_cache_batch_calls_r784(int kind,int device){return counts.at(kind).at(device).load();}
