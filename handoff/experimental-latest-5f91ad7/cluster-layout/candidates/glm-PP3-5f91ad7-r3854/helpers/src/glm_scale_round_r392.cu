#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstddef>
#include <cmath>
namespace {
__global__ void scale_round(const __nv_bfloat16*input,float*out,__nv_bfloat16*rounded,size_t count,float scale){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
 volatile float product=__bfloat162float(input[i])*scale;
 __nv_bfloat16 value=__float2bfloat16(product);
 out[i]=__bfloat162float(value);if(rounded)rounded[i]=value;
}
}
extern "C" int glm_scale_round_r392(const void*input,void*out,void*rounded,size_t count,float scale,void*stream){
 if(!input||!out||!count||count>size_t(1)<<31||!std::isfinite(scale)||scale<=0)return int(cudaErrorInvalidValue);
 scale_round<<<unsigned((count+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(input),static_cast<float*>(out),static_cast<__nv_bfloat16*>(rounded),count,scale);return int(cudaGetLastError());
}
