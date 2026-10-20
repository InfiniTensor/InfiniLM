#include "native/cuda/hygon/ops/indexed_kimi_delta_attention/kernel.h"
#include "native/cuda/hygon/ops/indexed_kimi_delta_attention/math.cuh"
#include <set>
namespace infini::ops::indexed_kda_detail {
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
std::vector<int64_t> Indices(const void* p,size_t n,bool i64,cudaStream_t stream){
  std::vector<int64_t> values(n);
  if(i64){Check(cudaMemcpyAsync(values.data(),p,n*8,cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));}
  else {std::vector<int32_t> small(n);Check(cudaMemcpyAsync(small.data(),p,n*4,cudaMemcpyDeviceToHost,stream));Check(cudaStreamSynchronize(stream));for(size_t i=0;i<n;++i)values[i]=small[i];}
  return values;
}
template<class T,class G> void Run(void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream){
  auto& s=l.strides;
  Check(cudaMemcpyAsync(snapshot,state,l.pool*l.heads*l.dim*l.dim*sizeof(T),cudaMemcpyDeviceToDevice,stream));
  kimiDeltaAttentionRecurrentCudaKernel<T,G><<<dim3(l.requests,l.heads,l.dim),256,(3*l.dim+256)*sizeof(float),stream>>>(
    static_cast<T*>(out),static_cast<T*>(snapshot),nullptr,static_cast<T*>(state),static_cast<const T*>(q),static_cast<const T*>(k),static_cast<const T*>(v),
    static_cast<const G*>(gate),static_cast<const G*>(beta),alog,bias,offsets,reads,writes,l.i64[0],l.i64[1],l.i64[2],l.norm,true,true,
    l.tokens,l.dim,l.pool,l.scale,l.lower,s[0][0],s[0][1],s[0][2],s[1][0],s[1][1],s[1][2],s[1][3],0,0,0,0,
    s[2][0],s[2][1],s[2][2],s[3][0],s[3][1],s[3][2],s[4][0],s[4][1],s[4][2],s[5][0],s[5][1],s[5][2],s[6][0],s[6][1],s[6][2],1,s[7][0]);
  Check(cudaGetLastError());
}
template<class T> void Gate(DataType g,void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream){
  switch(g){case DataType::kFloat32:Run<T,float>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kFloat16:Run<T,half>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kBFloat16:Run<T,__nv_bfloat16>(out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    default:throw std::invalid_argument("KDA gate dtype unsupported");}
}
void Launch(DataType data,DataType g,void* out,void* state,void* snapshot,const void* q,const void* k,const void* v,const void* gate,const void* beta,
    const float* alog,const float* bias,const void* offsets,const void* reads,const void* writes,Layout l,cudaStream_t stream){
  auto off=Indices(offsets,l.requests+1,l.i64[0],stream),ri=Indices(reads,l.requests,l.i64[1],stream),wi=Indices(writes,l.requests,l.i64[2],stream);
  std::set<int64_t> unique;
  if(off.front()!=0||off.back()!=int64_t(l.tokens))throw std::invalid_argument("KDA token offsets mismatch");
  for(size_t i=0;i<l.requests;++i)
    if(off[i]>off[i+1]||ri[i]<0||uint64_t(ri[i])>=l.pool||wi[i]<0||uint64_t(wi[i])>=l.pool||!unique.insert(wi[i]).second)
      throw std::invalid_argument("KDA invalid indices or duplicate state writes");
  switch(data){case DataType::kFloat32:Gate<float>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kFloat16:Gate<half>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    case DataType::kBFloat16:Gate<__nv_bfloat16>(g,out,state,snapshot,q,k,v,gate,beta,alog,bias,offsets,reads,writes,l,stream);break;
    default:throw std::invalid_argument("KDA data dtype unsupported");}
}
}
