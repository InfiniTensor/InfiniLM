#include "execution/owner_affinity_r2392.hpp"
extern "C" const char*modern_owner_affinity_report_r2392(){
 namespace a=infinilm::execution::affinity_r2392;static thread_local std::string result;std::lock_guard<std::mutex> lock(a::mutex);std::ostringstream s;s<<"{\"owners\":[";
 for(int d=0;d<8;++d){if(d)s<<',';s<<"{\"device\":"<<d<<",\"changes\":"<<a::changes[d].load()<<",\"verified_mask\":\""<<a::masks[d]<<"\"}";}s<<"]}";result=s.str();return result.c_str();
}
