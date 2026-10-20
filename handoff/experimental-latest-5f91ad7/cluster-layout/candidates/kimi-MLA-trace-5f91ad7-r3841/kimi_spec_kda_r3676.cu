#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include "native/cuda/hygon/ops/indexed_kimi_delta_attention/math.cuh"
#include "roundstep.cuh"
extern "C" int kda_rounding_r3676(int device,int rounded,int T,int R,int H,int D,int P,void*out,const void*initial,void*final,const void*q,const void*k,const void*v,const void*gate,const void*beta,const void*alog,const void*bias,const void*offsets,const void*reads,const void*writes,void*stream){
 int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||rounded<0||rounded>1||T<1||T>128||R<1||R>16||H!=12||D!=128||(P<2*R+1||P>129)||!out||!initial||!final||!q||!k||!v||!gate||!beta||!alog||!bias||!offsets||!reads||!writes||!stream)return -1;
 if(rounded){kimiDeltaAttentionRoundStepR3655<__nv_bfloat16,__nv_bfloat16><<<dim3(R,H,D),256,(3*D+256)*sizeof(float),static_cast<cudaStream_t>(stream)>>>(
  static_cast<__nv_bfloat16*>(out),const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(initial)),nullptr,static_cast<__nv_bfloat16*>(final),static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
  static_cast<const __nv_bfloat16*>(gate),static_cast<const __nv_bfloat16*>(beta),static_cast<const float*>(alog),static_cast<const float*>(bias),offsets,reads,writes,false,false,false,true,true,true,
  T,D,P,1.0f/sqrtf(float(D)),-5.0f,T*H*D,H*D,D,H*D*D,D*D,D,1,0,0,0,0,
  T*H*D,H*D,D,T*H*D,H*D,D,T*H*D,H*D,D,T*H*D,H*D,D,T*H,H,1,1,D);}else{kimiDeltaAttentionRecurrentCudaKernel<__nv_bfloat16,__nv_bfloat16><<<dim3(R,H,D),256,(3*D+256)*sizeof(float),static_cast<cudaStream_t>(stream)>>>(
  static_cast<__nv_bfloat16*>(out),const_cast<__nv_bfloat16*>(static_cast<const __nv_bfloat16*>(initial)),nullptr,static_cast<__nv_bfloat16*>(final),static_cast<const __nv_bfloat16*>(q),static_cast<const __nv_bfloat16*>(k),static_cast<const __nv_bfloat16*>(v),
  static_cast<const __nv_bfloat16*>(gate),static_cast<const __nv_bfloat16*>(beta),static_cast<const float*>(alog),static_cast<const float*>(bias),offsets,reads,writes,false,false,false,true,true,true,
  T,D,P,1.0f/sqrtf(float(D)),-5.0f,T*H*D,H*D,D,H*D*D,D*D,D,1,0,0,0,0,
  T*H*D,H*D,D,T*H*D,H*D,D,T*H*D,H*D,D,T*H*D,H*D,D,T*H,H,1,1,D);}
 return int(cudaGetLastError());
}
