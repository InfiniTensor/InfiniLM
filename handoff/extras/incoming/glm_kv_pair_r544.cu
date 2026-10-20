#include <cuda_runtime.h>
#include <atomic>
#include <cstddef>
#include <cstdint>
namespace {
std::atomic<unsigned long long> calls[8]{};
// The owner has already checked every ID and keeps all tensors alive to sync.
// Copy the BF16 storage bits, including signed zeros and NaN payloads.
__global__ void pair_copy(const uint16_t*k,const uint16_t*v,const int64_t*ids,
 uint16_t*ko,uint16_t*vo,size_t count,size_t row_elements){
 size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
 if(i>=count*row_elements)return;
 size_t src=size_t(ids[i/row_elements])*row_elements+i%row_elements;
 ko[i]=k[src];vo[i]=v[src];
}
}
extern "C" unsigned long long modern_glm_kv_pair_calls_r544(int device){
 return device>=0&&device<8?calls[device].load():0;
}
extern "C" int glm_kv_pair_r544(const void*k,const void*v,const void*ids,
 void*ko,void*vo,size_t count,size_t heads,void*stream){
 if(!k||!v||!ids||!ko||!vo||!count||!heads||heads>128||count>2048)return int(cudaErrorInvalidValue);
 int device=-1;auto status=cudaGetDevice(&device);if(status!=cudaSuccess)return int(status);
 size_t elements=count*heads*256;
 pair_copy<<<unsigned((elements+255)/256),256,0,static_cast<cudaStream_t>(stream)>>>(
  static_cast<const uint16_t*>(k),static_cast<const uint16_t*>(v),static_cast<const int64_t*>(ids),
  static_cast<uint16_t*>(ko),static_cast<uint16_t*>(vo),count,heads*256);
 status=cudaGetLastError();if(status==cudaSuccess&&device>=0&&device<8)++calls[device];return int(status);
}
