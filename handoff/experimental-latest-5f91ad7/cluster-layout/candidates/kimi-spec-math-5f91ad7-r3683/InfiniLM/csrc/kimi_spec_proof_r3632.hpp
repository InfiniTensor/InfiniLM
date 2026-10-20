#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace infinilm::models::kimi_native {
struct SpecProofR3632 {const void*owner;size_t capacity;std::vector<int32_t>kv,read,write;};
inline thread_local const SpecProofR3632*active_spec_proof_r3632=nullptr;
inline bool spec_enabled_r3632(){const char*p=std::getenv("INFINILM_KIMI_SPEC_R3632");return p&&std::string(p)=="1";}
class SpecScopeR3632 final {
 SpecProofR3632 proof_;
public:
 SpecScopeR3632(const void*owner,size_t capacity,const std::vector<int32_t>&kv,const std::vector<int32_t>&read,const std::vector<int32_t>&write):proof_{owner,capacity,kv,read,write}{
  if(!spec_enabled_r3632()||!owner||active_spec_proof_r3632||capacity<2||capacity>128||capacity%2||kv.empty()||kv.size()>capacity/2||read.size()!=kv.size()||write.size()!=kv.size())throw std::invalid_argument("3632 opt-in/owner/capacity/metadata count");
  std::set<int32_t>keys,slots;
  for(size_t i=0;i<kv.size();++i){
   const int32_t k=kv[i],a=read[i],b=write[i],half=int32_t(capacity/2);
   if(k<1||k>half||!keys.insert(k).second||!((a==k&&b==k+half)||(a==k+half&&b==k))||!slots.insert(a).second||!slots.insert(b).second)throw std::invalid_argument("3632 each logical KV row owns one nonalias state pair");
  }
  active_spec_proof_r3632=&proof_;
 }
 ~SpecScopeR3632()noexcept{active_spec_proof_r3632=nullptr;}
 SpecScopeR3632(const SpecScopeR3632&)=delete;
 SpecScopeR3632&operator=(const SpecScopeR3632&)=delete;
};
}
