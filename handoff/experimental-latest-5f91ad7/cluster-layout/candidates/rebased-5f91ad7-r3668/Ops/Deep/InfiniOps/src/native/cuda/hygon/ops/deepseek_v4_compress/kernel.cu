#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
namespace infini::ops::dsv4_compress_detail {
template<class T>__global__ void finite(const T*x,size_t n,int*status){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)if(!isfinite(float(x[i])))atomicOr(status,2);}
__global__ void metadata(const int64_t*pos,const int64_t*writes,const int64_t*extras,int*owners,int*status,size_t tokens,size_t rows,size_t ratio){
 for(size_t t=size_t(blockIdx.x)*blockDim.x+threadIdx.x;t<tokens;t+=size_t(gridDim.x)*blockDim.x){int64_t p=pos[t],group=writes[t],extra=extras[t];if(p<0||p>=1048576||group<0||size_t(group)>=rows/ratio||(ratio==4&&(extra<0||size_t(extra)>=rows/ratio))){atomicOr(status,1);continue;}atomicMax(owners+size_t(group)*ratio+size_t(p)%ratio,int(t));}
}
template<class State>__global__ void compress(__nv_bfloat16*out,const __nv_bfloat16*score,const float*ape,const State*state,const int64_t*positions,const int64_t*writes,const int64_t*extras,const int*status,size_t tokens,size_t dim,size_t ratio){
 if(*status)return;size_t token=blockIdx.x;if(token>=tokens)return;const int64_t pos=positions[token],group=writes[token];const size_t channels=ratio==4?4:2;
 for(size_t d=threadIdx.x;d<dim;d+=blockDim.x){if((pos+1)%int64_t(ratio)){out[token*dim+d]=__float2bfloat16_rn(0.f);continue;}float maximum=-FLT_MAX,numerator=0.f,denominator=0.f;
  for(size_t window=0;window<(ratio==4?2:1);++window){bool previous=ratio==4&&window==0;if(previous&&pos<7)continue;int64_t source_group=previous?extras[token]:group,begin=pos-(previous?7:int64_t(ratio)-1);size_t value_channel=ratio==4?(previous?0:1):0;
   for(size_t i=0;i<ratio;++i){int64_t expected=begin+int64_t(i),source=int64_t(token)-(pos-expected);bool in_batch=source>=0&&size_t(source)<tokens&&positions[source]==expected&&writes[source]==source_group;size_t value_index=in_batch?size_t(source)*channels*dim+value_channel*dim+d:(size_t(source_group)*ratio+i)*channels*dim+value_channel*dim+d;size_t score_index=value_index+(ratio==4?2:1)*dim;
    float value=in_batch?float(score[value_index]):float(state[value_index]);float weight=(in_batch?float(score[score_index]):float(state[score_index]))+ape[(window*ratio+i)*dim+d];float next=fmaxf(maximum,weight),old=expf(maximum-next),fresh=expf(weight-next);numerator=numerator*old+value*fresh;denominator=denominator*old+fresh;maximum=next;
   }
  }out[token*dim+d]=__float2bfloat16_rn(numerator/denominator);
 }
}
template<class State>__global__ void persist(const __nv_bfloat16*score,State*state,const int64_t*positions,const int64_t*writes,const int*owners,const int*status,size_t tokens,size_t dim,size_t ratio){
 if(*status)return;size_t token=blockIdx.x;if(token>=tokens)return;size_t slot=size_t(writes[token])*ratio+size_t(positions[token])%ratio;if(owners[slot]!=int(token))return;size_t channels=ratio==4?4:2;for(size_t j=threadIdx.x;j<channels*dim;j+=blockDim.x)state[slot*channels*dim+j]=State(score[token*channels*dim+j]);
}
void Launch(void*out,const void*score,const void*ape,void*state,bool state_f32,const void*positions,const void*writes,const void*extras,void*status,void*workspace,size_t tokens,size_t dim,size_t rows,size_t ratio,cudaStream_t stream){
 auto check=[](cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));};auto*flag=static_cast<int*>(status);auto*owners=static_cast<int*>(workspace);check(cudaMemsetAsync(flag,0,sizeof(int),stream));check(cudaMemsetAsync(owners,0xff,rows*sizeof(int),stream));size_t channels=ratio==4?4:2;auto grid=[](size_t n){return unsigned(std::min<size_t>((n+255)/256,4096));};
 metadata<<<grid(tokens),256,0,stream>>>(static_cast<const int64_t*>(positions),static_cast<const int64_t*>(writes),static_cast<const int64_t*>(extras),owners,flag,tokens,rows,ratio);
 finite<<<grid(tokens*channels*dim),256,0,stream>>>(static_cast<const __nv_bfloat16*>(score),tokens*channels*dim,flag);finite<<<grid((ratio==4?8:128)*dim),256,0,stream>>>(static_cast<const float*>(ape),(ratio==4?8:128)*dim,flag);
 if(state_f32){finite<<<grid(rows*channels*dim),256,0,stream>>>(static_cast<const float*>(state),rows*channels*dim,flag);compress<<<unsigned(tokens),128,0,stream>>>(static_cast<__nv_bfloat16*>(out),static_cast<const __nv_bfloat16*>(score),static_cast<const float*>(ape),static_cast<const float*>(state),static_cast<const int64_t*>(positions),static_cast<const int64_t*>(writes),static_cast<const int64_t*>(extras),flag,tokens,dim,ratio);persist<<<unsigned(tokens),256,0,stream>>>(static_cast<const __nv_bfloat16*>(score),static_cast<float*>(state),static_cast<const int64_t*>(positions),static_cast<const int64_t*>(writes),owners,flag,tokens,dim,ratio);}
 else{finite<<<grid(rows*channels*dim),256,0,stream>>>(static_cast<const __nv_bfloat16*>(state),rows*channels*dim,flag);compress<<<unsigned(tokens),128,0,stream>>>(static_cast<__nv_bfloat16*>(out),static_cast<const __nv_bfloat16*>(score),static_cast<const float*>(ape),static_cast<const __nv_bfloat16*>(state),static_cast<const int64_t*>(positions),static_cast<const int64_t*>(writes),static_cast<const int64_t*>(extras),flag,tokens,dim,ratio);persist<<<unsigned(tokens),256,0,stream>>>(static_cast<const __nv_bfloat16*>(score),static_cast<__nv_bfloat16*>(state),static_cast<const int64_t*>(positions),static_cast<const int64_t*>(writes),owners,flag,tokens,dim,ratio);}check(cudaGetLastError());
}
}
