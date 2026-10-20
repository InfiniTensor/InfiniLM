#include <cuda_runtime.h>
#include <cstddef>
#include <cmath>
namespace {
__global__ void causal_mask(float* logits,size_t total,size_t queries,size_t keys,size_t past){
 size_t index=blockIdx.x*blockDim.x+threadIdx.x;
 if(index<total && index%keys>past+(index/keys)%queries)logits[index]=-INFINITY;
}
}
extern "C" int modern_glm_dense_causal_mask_r25(void* data,size_t heads,size_t queries,size_t keys,size_t past,void* stream){
 if(!data||!heads||!queries||!keys||past>keys||queries>keys-past)return int(cudaErrorInvalidValue);
 size_t total=heads*queries*keys;
 causal_mask<<<(total+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(data),total,queries,keys,past);
 return int(cudaGetLastError());
}
