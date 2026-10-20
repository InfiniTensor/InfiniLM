#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <limits>
namespace {
__global__ void gather(float*out,const uint16_t*swa,const uint16_t*compressed,
 const int64_t*ids,size_t count,size_t swa_rows,int64_t s0,int64_t s1,int64_t c0,int64_t c1){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=count*512)return;
 size_t row=i/512,col=i%512;int64_t id=ids[row];
 const uint16_t*bank=id<int64_t(swa_rows)?swa:compressed;
 int64_t offset=id<int64_t(swa_rows)?id*s0+int64_t(col)*s1:(id-int64_t(swa_rows))*c0+int64_t(col)*c1;
 union {uint32_t bits;float value;}v;v.bits=uint32_t(bank[offset])<<16;
 out[i]=v.value;
}
}
// IDs have already been validated on the owner thread by latent_selection_r356.
extern "C" int deep_packed_gather_r374(void*out,const void*swa,const void*compressed,
 const void*ids,size_t count,size_t swa_rows,int64_t s0,int64_t s1,int64_t c0,int64_t c1,void*stream){
 if(!out||!swa||!compressed||!ids||!count||!swa_rows||count>size_t(UINT32_MAX)*256/512||swa_rows>size_t(INT64_MAX))return int(cudaErrorInvalidValue);
 gather<<<unsigned((count*512+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(out),static_cast<const uint16_t*>(swa),static_cast<const uint16_t*>(compressed),static_cast<const int64_t*>(ids),count,swa_rows,s0,s1,c0,c1);
 return int(cudaGetLastError());
}
