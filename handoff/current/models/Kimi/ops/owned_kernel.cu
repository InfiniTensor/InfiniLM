#include "owned_provider.hpp"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>
namespace owned_kda_math_r1183 {
#include "owned_math.cuh"
}
#include <set>
namespace infini::ops::indexed_kda_detail {
void CheckR1183(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T,class G> void RunR1183(void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream){
  auto& s=l.strides;
  void*initial=state;
  if(!owned_kda_r1183::allow_identity||!owned_kda_r1183::current_frame->identity){CheckR1183(cudaMemcpyAsync(snapshot,state,l.pool*l.heads*l.dim*l.dim*sizeof(T),cudaMemcpyDeviceToDevice,stream));initial=snapshot;}

  owned_kda_math_r1183::kimiDeltaAttentionRecurrentCudaKernel<T,G><<<dim3(l.requests,l.heads,l.dim),256,(3*l.dim+256)*sizeof(float),stream>>>(
    static_cast<T*>(out),static_cast<T*>(initial),nullptr,static_cast<T*>(state),static_cast<const T*>(q),static_cast<const T*>(k),static_cast<const T*>(v),
    static_cast<const G*>(gate),static_cast<const G*>(beta),alog,bias,offsets,reads,writes,l.i64[0],l.i64[1],l.i64[2],l.norm,true,true,
    l.tokens,l.dim,l.pool,l.scale,l.lower,s[0][0],s[0][1],s[0][2],s[1][0],s[1][1],s[1][2],s[1][3],0,0,0,0,
    s[2][0],s[2][1],s[2][2],s[3][0],s[3][1],s[3][2],s[4][0],s[4][1],s[4][2],s[5][0],s[5][1],s[5][2],s[6][0],s[6][1],s[6][2],1,s[7][0]);
  CheckR1183(cudaGetLastError());
}
template<class T> void GateR1183(DataType g,void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream){
  switch(g){case DataType::kFloat32:RunR1183<T,float>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kFloat16:RunR1183<T,half>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kBFloat16:RunR1183<T,__nv_bfloat16>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    default:throw std::invalid_argument("KDA gate dtype unsupported");}
}
void LaunchR1183(DataType data,DataType g,void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream){
  const auto*frame=owned_kda_r1183::current_frame;if(!frame)throw std::logic_error("1183missing scoped metadata");
  frame->require(offsets,reads,writes,l.requests,l.tokens,l.pool,stream,l.i64);
  switch(data){case DataType::kFloat32:GateR1183<float>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kFloat16:GateR1183<half>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kBFloat16:GateR1183<__nv_bfloat16>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    default:throw std::invalid_argument("KDA data dtype unsupported");}
}
}
