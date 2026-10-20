#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cmath>

namespace {
constexpr int B=256,MAX_KEYS=2048;
__device__ float widen(uint16_t x){union{uint32_t u;float f;}v;v.u=uint32_t(x)<<16;return v.f;}
__device__ uint16_t narrow(float x){union{uint32_t u;float f;}v;v.f=x;if((v.u&0x7fffffff)>0x7f800000)return uint16_t((v.u>>16)|0x40);return uint16_t((v.u+0x7fff+((v.u>>16)&1))>>16);}
__device__ float warp_sum(float x){
 for(int offset=16;offset;offset/=2){
#if defined(__HIP_PLATFORM_AMD__)
  x+=__shfl_down(x,offset,32);
#else
  x+=__shfl_down_sync(0xffffffffu,x,offset,32);
#endif
 }return x;
}
__global__ void validate(const int64_t*si,const int64_t*ci,const int32_t*lengths,
 size_t sw,size_t cw,size_t sr,size_t cr,int*error){
 const size_t t=blockIdx.x;const int lane=threadIdx.x;__shared__ int ns,nc,bad;__shared__ int64_t ids[MAX_KEYS];
 if(lane==0){ns=lengths[2*t];nc=lengths[2*t+1];bad=ns<0||nc<0||size_t(ns)>sw||size_t(nc)>cw;if(bad)atomicMax(error,1);}
 __syncthreads();if(bad)return;const int n=ns+nc;
 for(int j=lane;j<n;j+=B){bool compressed=j>=ns;int64_t id=compressed?ci[t*cw+size_t(j-ns)]:si[t*sw+size_t(j)];size_t rows=compressed?cr:sr;
  if(id<0||uint64_t(id)>=rows){atomicMax(error,2);ids[j]=INT64_MIN;}else ids[j]=id+(compressed?int64_t(sr):0);
 }__syncthreads();
 // Duplicates are invalid in the existing R356 contract; never deduplicate silently.
 for(int j=lane;j<n;j+=B)for(int k=0;k<j;++k)if(ids[j]==ids[k]){atomicMax(error,3);break;}
}
__global__ void attention_parts(float*work,const uint16_t*q,const uint16_t*swa,const uint16_t*comp,const float*sink,
 const int64_t*si,const int64_t*ci,const int32_t*lengths,size_t heads,size_t sw,size_t cw,size_t parts,
 int64_t q0,int64_t q1,int64_t q2,int64_t s0,int64_t s1,int64_t c0,int64_t c1,int64_t sink_stride){
 const int t=blockIdx.x,h=blockIdx.y,part=blockIdx.z,lane=threadIdx.x,warp=lane/32,wl=lane%32;
 const int ns=lengths[2*t],nc=lengths[2*t+1],n=ns+nc,begin=part*64,count=min(64,max(0,n+1-begin));
 float*dest=work+((size_t(t)*heads+h)*parts+part)*514;
 if(count==0){for(int d=lane;d<512;d+=B)dest[d]=0;if(lane==0){dest[512]=-INFINITY;dest[513]=0;}return;}
 __shared__ float query[512],scores[64],reduce[B];__shared__ float maximum,total_weight;
 for(int d=lane;d<512;d+=B)query[d]=widen(q[int64_t(t)*q0+int64_t(h)*q1+int64_t(d)*q2]);__syncthreads();
 for(int j=warp;j<count;j+=B/32){int global=begin+j;if(global==n){if(wl==0)scores[j]=sink[int64_t(h)*sink_stride];}else{
  bool compressed=global>=ns;int64_t id=compressed?ci[size_t(t)*cw+size_t(global-ns)]:si[size_t(t)*sw+size_t(global)];const uint16_t*bank=compressed?comp:swa;int64_t row=id*(compressed?c0:s0),stride=compressed?c1:s1;float sum=0;
  for(int d=wl;d<512;d+=32)sum+=query[d]*widen(bank[row+int64_t(d)*stride]);sum=warp_sum(sum);if(wl==0)scores[j]=sum*0.04419417382415922f;
 }}__syncthreads();
 float m=-INFINITY;for(int j=lane;j<count;j+=B)m=fmaxf(m,scores[j]);reduce[lane]=m;__syncthreads();
 for(int stride=B/2;stride;stride/=2){if(lane<stride)reduce[lane]=fmaxf(reduce[lane],reduce[lane+stride]);__syncthreads();}
 if(lane==0)maximum=reduce[0];__syncthreads();float sum_weights=0;
 for(int j=lane;j<count;j+=B){float p=expf(scores[j]-maximum);scores[j]=p;sum_weights+=p;}reduce[lane]=sum_weights;__syncthreads();
 for(int stride=B/2;stride;stride/=2){if(lane<stride)reduce[lane]+=reduce[lane+stride];__syncthreads();}
 if(lane==0){total_weight=reduce[0];dest[512]=maximum;dest[513]=total_weight;}__syncthreads();
 for(int d=lane;d<512;d+=B){float sum=0;for(int j=0;j<count;++j){int global=begin+j;if(global==n)continue;bool compressed=global>=ns;int64_t id=compressed?ci[size_t(t)*cw+size_t(global-ns)]:si[size_t(t)*sw+size_t(global)];const uint16_t*bank=compressed?comp:swa;int64_t offset=id*(compressed?c0:s0)+int64_t(d)*(compressed?c1:s1);sum+=scores[j]*widen(bank[offset]);}dest[d]=sum;}
}
__global__ void attention_merge(uint16_t*out,const float*work,size_t heads,size_t parts){
 const size_t row=size_t(blockIdx.x)*heads+blockIdx.y;const int lane=threadIdx.x;const float*source=work+row*parts*514;
 __shared__ float scales[33],denominator;
 if(lane==0){float maximum=-INFINITY;for(size_t p=0;p<parts;++p)maximum=fmaxf(maximum,source[p*514+512]);float total=0;
  for(size_t p=0;p<parts;++p){float weight=source[p*514+513];float scale=weight>0?expf(source[p*514+512]-maximum):0;scales[p]=scale;total+=weight*scale;}denominator=total;
 }__syncthreads();
 for(int d=lane;d<512;d+=B){float sum=0;for(size_t p=0;p<parts;++p)sum+=source[p*514+d]*scales[p];out[row*512+d]=narrow(sum/denominator);}
}

}
extern "C" int deep_latent_validate_r681(const void*si,const void*ci,const void*lengths,size_t tokens,size_t sw,size_t cw,size_t sr,size_t cr,void*error,void*stream){
 if(!si||!ci||!lengths||!error||!tokens||tokens>64||!sw||!cw||sw>MAX_KEYS||cw>MAX_KEYS-sw||!sr||!cr||sr>size_t(INT64_MAX)||cr>size_t(INT64_MAX)-sr)return int(cudaErrorInvalidValue);
 validate<<<unsigned(tokens),B,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const int64_t*>(si),static_cast<const int64_t*>(ci),static_cast<const int32_t*>(lengths),sw,cw,sr,cr,static_cast<int*>(error));return int(cudaGetLastError());
}
extern "C" int deep_latent_compute_r681(void*out,void*workspace,size_t parts,const void*q,const void*swa,const void*comp,const void*sink,const void*si,const void*ci,const void*lengths,size_t tokens,size_t heads,size_t sw,size_t cw,int64_t q0,int64_t q1,int64_t q2,int64_t s0,int64_t s1,int64_t c0,int64_t c1,int64_t sink_stride,void*stream){
 if(!out||!workspace||!q||!swa||!comp||!sink||!si||!ci||!lengths||!tokens||tokens>64||!heads||heads>128||!sw||!cw||sw>MAX_KEYS||cw>MAX_KEYS-sw||parts!=(sw+cw+64)/64||parts>33)return int(cudaErrorInvalidValue);
 attention_parts<<<dim3(unsigned(tokens),unsigned(heads),unsigned(parts)),B,0,static_cast<cudaStream_t>(stream)>>>(static_cast<float*>(workspace),static_cast<const uint16_t*>(q),static_cast<const uint16_t*>(swa),static_cast<const uint16_t*>(comp),static_cast<const float*>(sink),static_cast<const int64_t*>(si),static_cast<const int64_t*>(ci),static_cast<const int32_t*>(lengths),heads,sw,cw,parts,q0,q1,q2,s0,s1,c0,c1,sink_stride);
 auto status=cudaGetLastError();if(status!=cudaSuccess)return int(status);
 attention_merge<<<dim3(unsigned(tokens),unsigned(heads)),B,0,static_cast<cudaStream_t>(stream)>>>(static_cast<uint16_t*>(out),static_cast<const float*>(workspace),heads,parts);return int(cudaGetLastError());
}
