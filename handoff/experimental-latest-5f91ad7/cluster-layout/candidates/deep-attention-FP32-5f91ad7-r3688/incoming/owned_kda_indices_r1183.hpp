#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
namespace owned_kda_r1183 {
inline std::atomic<uint64_t> sequence{0};
inline uint64_t next_id(){auto n=sequence.fetch_add(1,std::memory_order_relaxed)+1;if(!n)std::terminate();return n;}
inline void check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
// The buffer belongs to its context, outlives cached graphs, and is never
// exposed by the C API. Each epoch copies and validates new CPU values before
// uploading. Reusing a GPU address never substitutes for validating an epoch.
struct Frame {
 const void*owner;int device;cudaStream_t stream;std::thread::id thread;
 uint64_t id;size_t requests,tokens,pool;bool active=true,identity;
 void*buffer;std::vector<int64_t>values;
 Frame(const void*key,int dev,cudaStream_t s,void*gpu,size_t nr,size_t nt,size_t np,
       const int64_t*off,const int64_t*reads,const int64_t*writes)
  :owner(key),device(dev),stream(s),thread(std::this_thread::get_id()),id(next_id()),requests(nr),tokens(nt),pool(np),buffer(gpu){
  if(!key||!gpu||!off||!reads||!writes||!nr||nr>128||!nt||nt>65536||!np||np>65536)
   throw std::invalid_argument("1183frame geometry/pointers");
  values.reserve(3*nr+1);values.insert(values.end(),off,off+nr+1);values.insert(values.end(),reads,reads+nr);values.insert(values.end(),writes,writes+nr);
  if(values.front()!=0||values[nr]!=int64_t(nt))throw std::invalid_argument("1183offset endpoints");
  std::set<int64_t>unique;identity=true;
  for(size_t i=0;i<nr;++i){auto r=values[nr+1+i],w=values[2*nr+1+i];
   if(values[i]<0||values[i]>values[i+1]||values[i+1]>int64_t(nt)||r<0||uint64_t(r)>=np||w<0||uint64_t(w)>=np||!unique.insert(w).second)
    throw std::invalid_argument("1183invalid offsets/rows/duplicate writes");
   identity&=r==w;
  }
 }
 void own()const{if(!active||std::this_thread::get_id()!=thread)throw std::logic_error("1183inactive/foreign frame");int current=-1;check(cudaGetDevice(&current));if(current!=device)throw std::logic_error("1183foreign device");}
 const int64_t*offsets()const{return static_cast<const int64_t*>(buffer);}
 const int64_t*reads()const{return offsets()+requests+1;}
 const int64_t*writes()const{return reads()+requests;}
 void upload(){own();check(cudaMemcpyAsync(buffer,values.data(),values.size()*8,cudaMemcpyHostToDevice,stream));}
 void require(const void*off,const void*ri,const void*wi,size_t nr,size_t nt,size_t np,cudaStream_t s,const bool*i64)const{
  own();if(off!=offsets()||ri!=reads()||wi!=writes()||nr!=requests||nt!=tokens||np!=pool||s!=stream||!i64[0]||!i64[1]||!i64[2])
   throw std::logic_error("1183frame binding/geometry differs");
 }
};
inline thread_local const Frame*current_frame=nullptr;
inline thread_local bool allow_identity=false;
struct Scope {
 Scope(const Frame&f,bool fast){f.own();if(current_frame)throw std::logic_error("1183nested frame invocation");current_frame=&f;allow_identity=fast;}
 ~Scope(){current_frame=nullptr;allow_identity=false;}
 Scope(const Scope&)=delete;Scope&operator=(const Scope&)=delete;
};
}
