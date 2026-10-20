#include "kimi_owned_indices_raw_r1213.hpp"
#include "owned_provider.hpp"
#include "native/cuda/hygon/ops/causal_conv1d/kernel.h"
#include "owned_conv_provider.hpp"
#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
using namespace infini::ops;
namespace {
thread_local std::string last_error;
std::array<std::array<std::atomic<uint64_t>,12>,8> counters{};
using owned_kda_r1183::check;
struct Request {
 int device,mode;cudaStream_t stream;std::thread::id owner;
 void *workspace;size_t bytes;std::unique_ptr<owned_kda_r1183::Frame> frame;
};
void require_owner(Request &r){
 if(r.owner!=std::this_thread::get_id())throw std::logic_error("1213foreign request thread");
 int current=-1;check(cudaGetDevice(&current));if(current!=r.device)throw std::logic_error("1213foreign request device");
 if(!r.frame)throw std::logic_error("1213request epoch absent");r.frame->own();
 cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(r.stream,&status));if(status!=cudaStreamCaptureStatusNone)throw std::logic_error("1213Runtime eager adapter refuses graph capture");
}
std::array<Tensor,3> metadata(Request&r){
 Device device{Device::Type::kHygon,r.device};auto&f=*r.frame;
 return {Tensor(reinterpret_cast<std::byte*>(const_cast<int64_t*>(f.offsets())),{f.requests+1},DataType::kInt64,device),Tensor(reinterpret_cast<std::byte*>(const_cast<int64_t*>(f.reads())),{f.requests},DataType::kInt64,device),Tensor(reinterpret_cast<std::byte*>(const_cast<int64_t*>(f.writes())),{f.requests},DataType::kInt64,device)};
}
template<class Op>void bind(Request&r,Op&op){op.set_stream(r.stream);op.set_workspace(r.workspace);op.set_workspace_size_in_bytes(r.bytes);}
}
extern "C" const char *modern_kimi_owned_error_r1213(){return last_error.c_str();}
extern "C" void *modern_kimi_owned_begin_r1213(int device,void *stream,void *metadata_buffer,size_t metadata_bytes,void *workspace,size_t workspace_bytes,size_t requests,size_t tokens,size_t pool,const int64_t *offsets,const int64_t *reads,const int64_t *writes,int mode){
 std::unique_ptr<Request>r;try{
  if(device<0||device>=8||mode<0||mode>2||!metadata_buffer||!workspace||reinterpret_cast<uintptr_t>(workspace)%256||!workspace_bytes||!requests||requests>32||requests!=tokens||metadata_bytes<(3*requests+1)*sizeof(int64_t))throw std::invalid_argument("1213owned decode geometry/storage/mode");
  int current=-1;check(cudaGetDevice(&current));if(current!=device)throw std::invalid_argument("1213begin device differs");cudaStreamCaptureStatus status;check(cudaStreamIsCapturing(static_cast<cudaStream_t>(stream),&status));if(status!=cudaStreamCaptureStatusNone)throw std::invalid_argument("1213begin during graph capture");
  r=std::make_unique<Request>();r->device=device;r->mode=mode;r->stream=static_cast<cudaStream_t>(stream);r->owner=std::this_thread::get_id();r->workspace=workspace;r->bytes=workspace_bytes;
  r->frame=std::make_unique<owned_kda_r1183::Frame>(r.get(),device,r->stream,metadata_buffer,requests,tokens,pool,offsets,reads,writes);
  // Reject non-decode offsets before upload, even if their sum happens to match.
  for(size_t i=0;i<=requests;++i)if(r->frame->values[i]!=int64_t(i))throw std::invalid_argument("1213one token per request required");
  r->frame->upload();++counters[device][0];++counters[device][2];return r.release();
 }catch(const std::exception&e){last_error=e.what();if(r&&r->frame&&cudaStreamSynchronize(r->stream)!=cudaSuccess)std::terminate();return nullptr;}
}
extern "C" int modern_kimi_owned_end_r1213(void*raw){try{
 if(!raw)throw std::invalid_argument("1213null request");auto&r=*static_cast<Request*>(raw);require_owner(r);
 // One completed-request boundary. Frame's CPU upload source and both borrowed
 // Runtime allocations remain alive until this drain has succeeded.
 check(cudaStreamSynchronize(r.stream));++counters[r.device][9];r.frame->active=false;++counters[r.device][1];delete &r;return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_kimi_owned_kda_raw_r1213(void*raw,const Tensor*v,float scale,float lower,bool normalize){try{
 if(!raw||!v)throw std::invalid_argument("1213null KDA operands");auto&r=*static_cast<Request*>(raw);require_owner(r);auto m=metadata(r);
 auto call=[&](IndexedKimiDeltaAttention&op){bind(r,op);op(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],m[0],m[1],m[2],scale,lower,normalize);};
 if(!r.mode){Operator<IndexedKimiDeltaAttention,Device::Type::kHygon>op(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],m[0],m[1],m[2],scale,lower,normalize);call(op);++counters[r.device][7];}
 else{OwnedKdaR1183 op(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8],m[0],m[1],m[2],scale,lower,normalize);owned_kda_r1183::Scope scope(*r.frame,r.mode==2);call(op);++counters[r.device][3];if(r.mode==2&&r.frame->identity)++counters[r.device][5];}
 return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_kimi_owned_conv_raw_r1213(void*raw,const Tensor*v){try{
 if(!raw||!v)throw std::invalid_argument("1213null Conv operands");auto&r=*static_cast<Request*>(raw);require_owner(r);auto m=metadata(r);
 auto call=[&](CausalConv1d&op){bind(r,op);op(v[2],v[3],v[1],std::nullopt,std::optional<Tensor>{m[0]},std::optional<Tensor>{m[1]},std::optional<Tensor>{m[2]},v[0],std::nullopt);};
 if(!r.mode){Operator<CausalConv1d,Device::Type::kHygon>op(v[2],v[3],v[1],std::nullopt,std::optional<Tensor>{m[0]},std::optional<Tensor>{m[1]},std::optional<Tensor>{m[2]},v[0],std::nullopt);call(op);++counters[r.device][8];}
 else{OwnedConvR1201 op(v[2],v[3],v[1],std::nullopt,std::optional<Tensor>{m[0]},std::optional<Tensor>{m[1]},std::optional<Tensor>{m[2]},v[0],std::nullopt);owned_kda_r1183::Scope scope(*r.frame,r.mode==2);call(op);++counters[r.device][4];if(r.mode==2&&r.frame->identity)++counters[r.device][6];}
 return 0;
 }catch(const std::exception&e){last_error=e.what();return 1;}}
extern "C" int modern_kimi_owned_stats_r1213(int device,uint64_t*out){if(device<0||device>=8||!out)return 1;for(size_t i=0;i<12;++i)out[i]=counters[device][i].load();return 0;}
