#include <array>
#include <atomic>
#include <cstdint>
namespace {std::array<std::atomic<uint64_t>,8>calls{};}
extern "C" void modern_glm_fp8_count_r780(int d){calls.at(d).fetch_add(1,std::memory_order_relaxed);}
extern "C" uint64_t modern_glm_fp8_calls_r780(int d){return calls.at(d).load();}
