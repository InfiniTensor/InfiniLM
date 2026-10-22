#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "base/fused_moe_mxfp4.h"
#include "native/cuda/ops/fused_moe_mxfp4/kernel.cuh"
#include "kimi_mxfp4_static_r326.cuh"
#include "kimi_tiled_w13_r1858.cuh"
extern "C" int modern_kimi_tiled_batch_r4391(int device,void*out,void*activation,const void*input,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,size_t experts,size_t tokens,void*stream){
 using namespace infini::ops::mxfp4_detail;using T=__nv_bfloat16;int current=-1;
 if(cudaGetDevice(&current)!=cudaSuccess||current!=device||!out||!activation||!input||!ids||!weights||!w13||!s13||!w2||!s2||experts<16||experts>896||!tokens||tokens>16)return int(cudaErrorInvalidValue);
 auto s=static_cast<cudaStream_t>(stream);auto*i=static_cast<const int32_t*>(ids);
 tiledW13R1858<T,4,64><<<1536*tokens,256,8192,s>>>(static_cast<T*>(activation),static_cast<const T*>(input),i,static_cast<const uint8_t*>(w13),static_cast<const uint8_t*>(s13),experts,16*tokens);
 auto code=cudaGetLastError();if(code!=cudaSuccess)return int(code);
 staticW2R326U<<<3584*tokens,256,0,s>>>(static_cast<T*>(out),static_cast<const T*>(activation),i,static_cast<const float*>(weights),static_cast<const uint8_t*>(w2),static_cast<const uint8_t*>(s2),tokens,size_t{16},experts,size_t{3584},size_t{384},size_t{0});
 return int(cudaGetLastError());
}
