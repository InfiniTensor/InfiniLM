#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace{
__global__ void scatter(float*out,const uint16_t*local,const int64_t*ids,const float*weights,size_t rows,size_t routes){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=rows*4096)return;size_t row=i/4096,h=i%4096;int64_t route=ids[row];if(route<0||size_t(route)>=routes)return;
 union{uint32_t bits;float value;}v;v.bits=uint32_t(local[i])<<16;float product;float weight=weights[size_t(route)];asm volatile("v_mul_f32 %0, %1, %2" : "=v"(product) : "v"(v.value),"v"(weight));out[size_t(route)*4096+h]=product;
}
}
extern "C" int deep_prefill_scatter_r194(void*out,const void*local,const void*ids,const void*weights,size_t rows,size_t routes,void*stream){
 if(!out||!local||!ids||!weights||!rows||rows>routes||routes>4096*6||routes%6)return int(cudaErrorInvalidValue);
 scatter<<<unsigned((rows*4096+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(out),static_cast<const uint16_t*>(local),static_cast<const int64_t*>(ids),static_cast<const float*>(weights),rows,routes);return int(cudaGetLastError());
}
