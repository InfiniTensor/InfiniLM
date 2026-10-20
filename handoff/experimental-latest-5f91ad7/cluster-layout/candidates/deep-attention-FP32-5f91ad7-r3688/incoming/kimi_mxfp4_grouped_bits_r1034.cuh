#pragma once
// Share weight decode across routes with the same expert, retaining each
// original 256-lane accumulation and binary reduction tree independently.
namespace infini::ops::mxfp4_detail {
template<int Tile>
__global__ void bitsGroupR1034(const int32_t*ids,int32_t*groups,unsigned*count,size_t routes){
 size_t first=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(first>=routes)return;
 int32_t expert=ids[first];for(size_t p=0;p<first;++p)if(ids[p]==expert)return;
 unsigned matches=0;for(size_t p=first;p<routes;++p)matches+=ids[p]==expert;
 unsigned base=atomicAdd(count,(matches+Tile-1)/Tile),j=0;
 for(size_t p=first;p<routes;++p)if(ids[p]==expert){groups[(base+j/Tile)*Tile+j%Tile]=int32_t(p);++j;}
 for(;j%Tile;++j)groups[(base+j/Tile)*Tile+j%Tile]=-1;
}
template<class T,int Tile>
__global__ void bitsGroupedW13R1034(T*activated,const T*input,const int32_t*ids,
 const uint8_t*packed,const uint8_t*scales,const int32_t*groups,const unsigned*count,size_t experts,size_t route_count){
 constexpr size_t hidden=3584,intermediate=384,topk=16,pw=hidden/2,sw=hidden/32;
 size_t group=blockIdx.x/intermediate,i=blockIdx.x%intermediate;
 // Sparse routing may offer too little reuse to amortize extra accumulators.
 // All blocks see the same completed GPU grouping counter and choose one path.
 if(size_t(*count)*5>route_count*(Tile==4?3:4)){
  size_t route=group;if(route>=route_count)return;int32_t e=ids[route];
  if(e<0||size_t(e)>=experts){if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(0.f);return;}
  size_t gr=size_t(e)*2*intermediate+i,ur=gr+intermediate;const T*x=input+(route/topk)*hidden;float sums[2]={};
  for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){float gl,gh,ul,uh;decodePairBitsR1034(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);decodePairBitsR1034(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);float a=mxfp4Load(x,2*p),b=mxfp4Load(x,2*p+1);sums[0]+=a*gl+b*gh;sums[1]+=a*ul+b*uh;}
  extern __shared__ float fallback_scratch[];mxfp4BlockReduce(sums,fallback_scratch);if(threadIdx.x==0)activated[route*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[0],sums[1],size_t{2}));return;
 }
 if(group>=*count)return;
 int routes[Tile];for(int row=0;row<Tile;++row)routes[row]=groups[group*Tile+row];
 int32_t expert=ids[routes[0]];
 if(expert<0||size_t(expert)>=experts){if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(0.f);return;}
 size_t gr=size_t(expert)*2*intermediate+i,ur=gr+intermediate;float sums[2*Tile]={};
 for(size_t p=threadIdx.x;p<pw;p+=blockDim.x){
  float gl,gh,ul,uh;decodePairBitsR1034(packed[gr*pw+p],scales[gr*sw+p/16],gl,gh);decodePairBitsR1034(packed[ur*pw+p],scales[ur*sw+p/16],ul,uh);
  #pragma unroll
  for(int row=0;row<Tile;++row)if(routes[row]>=0){const T*x=input+(size_t(routes[row])/topk)*hidden;float a=mxfp4Load(x,p*2),b=mxfp4Load(x,p*2+1);sums[2*row]+=a*gl+b*gh;sums[2*row+1]+=a*ul+b*uh;}
 }
 extern __shared__ float scratch[];mxfp4BlockReduce(sums,scratch);
 if(threadIdx.x==0)for(int row=0;row<Tile;++row)if(routes[row]>=0)activated[size_t(routes[row])*intermediate+i]=mxfp4Store<T>(fusedMoeMxfp4Activate(sums[2*row],sums[2*row+1],size_t{2}));
}
}
