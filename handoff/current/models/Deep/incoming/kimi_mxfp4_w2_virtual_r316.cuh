#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
namespace infini::ops::mxfp4_detail {
template<class T,int Width,int Outputs>
__global__ void W2VirtualR316(T*out,const T*activated,const int32_t*ids,const float*weights,
 const uint8_t*packed,const uint8_t*scales,size_t tokens,size_t topk,size_t experts,size_t hidden,size_t intermediate,size_t offset){
 constexpr int Virtual=32/Width,ThreadsPerOutput=16*Width;
 const int output_group=threadIdx.x/ThreadsPerOutput;
 const int within=threadIdx.x%ThreadsPerOutput,route_id=within/Width,lane=within%Width;
 const size_t index=offset+size_t(blockIdx.x)*Outputs+output_group,token=index/hidden,channel=index%hidden;
 const bool valid=token<tokens;
 float sums[Virtual]={};int32_t expert=-1;
 if(valid&&size_t(route_id)<topk){expert=ids[token*topk+route_id];if(expert>=0&&size_t(expert)<experts){
  const size_t row=size_t(expert)*hidden+channel,base=(token*topk+route_id)*intermediate;
  for(size_t first=lane;first<intermediate/2;first+=32){
   #pragma unroll
   for(int v=0;v<Virtual;++v){size_t p=first+size_t(v)*Width;if(p<intermediate/2){float low,high;mxfp4DecodePair(packed[row*(intermediate/2)+p],scales[row*(intermediate/32)+p/16],low,high);
    sums[v]+=mxfp4Load(activated,base+2*p)*low+mxfp4Load(activated,base+2*p+1)*high;
   }}
  }
 }}
 // Fold high virtual lane bits first: exactly the original width32 add tree.
 #pragma unroll
 for(int shift=Virtual/2;shift>0;shift/=2){
  #pragma unroll
  for(int v=0;v<shift;++v)sums[v]+=sums[v+shift];
 }
 float sum=sums[0];
 #pragma unroll
 for(int shift=Width/2;shift>0;shift/=2){
#if defined(__HIPCC__)
  sum+=__shfl_down(sum,shift,Width);
#else
  sum+=__shfl_down_sync(0xffffffffu,sum,shift,Width);
#endif
 }
 __shared__ float route_sums[Outputs][16];if(lane==0)route_sums[output_group][route_id]=sum;
 __syncthreads();
 if(within==0&&valid){float value=0.f;for(size_t route=0;route<topk;++route){int32_t id=ids[token*topk+route];if(id>=0&&size_t(id)<experts)value+=weights[token*topk+route]*route_sums[output_group][route];}out[index]=mxfp4Store<T>(value);}
}
template<class T>void launchW2R316(T*out,const T*activated,const int32_t*ids,const float*weights,const uint8_t*packed,const uint8_t*scales,size_t tokens,size_t topk,size_t experts,size_t hidden,size_t intermediate,int mode,cudaStream_t stream){
 const size_t total=tokens*hidden;
 constexpr size_t max_blocks=std::numeric_limits<uint32_t>::max()/512;
 int outputs=mode==3?2:mode==4?4:1;
 for(size_t offset=0;offset<total;offset+=max_blocks*size_t(outputs)){
  size_t blocks=std::min(max_blocks,(total-offset+outputs-1)/outputs);
  if(mode==0)fusedMoeMxfp4W2Routes16Kernel<<<unsigned(blocks),256,0,stream>>>(out,activated,ids,weights,packed,scales,tokens,topk,experts,hidden,intermediate,offset);
  else if(mode==1)W2VirtualR316<T,32,1><<<unsigned(blocks),512,0,stream>>>(out,activated,ids,weights,packed,scales,tokens,topk,experts,hidden,intermediate,offset);
  else if(mode==2)W2VirtualR316<T,16,1><<<unsigned(blocks),256,0,stream>>>(out,activated,ids,weights,packed,scales,tokens,topk,experts,hidden,intermediate,offset);
  else if(mode==3)W2VirtualR316<T,8,2><<<unsigned(blocks),256,0,stream>>>(out,activated,ids,weights,packed,scales,tokens,topk,experts,hidden,intermediate,offset);
  else if(mode==4)W2VirtualR316<T,4,4><<<unsigned(blocks),256,0,stream>>>(out,activated,ids,weights,packed,scales,tokens,topk,experts,hidden,intermediate,offset);
 }
}
}
