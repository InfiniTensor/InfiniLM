#include "cpu_strided_copy_r2515.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>
using Shape=std::vector<std::size_t>;
using Strides=std::vector<std::ptrdiff_t>;
void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
void original(std::byte*d,const std::byte*s,const Shape&shape,const Strides&ds,
              const Strides&ss,std::size_t bytes,std::size_t dim=0){
 if(dim==shape.size()){std::memmove(d,s,bytes);return;}
 for(std::size_t i=0;i<shape[dim];++i)original(d+std::ptrdiff_t(i)*ds[dim]*std::ptrdiff_t(bytes),s+std::ptrdiff_t(i)*ss[dim]*std::ptrdiff_t(bytes),shape,ds,ss,bytes,dim+1);
}
struct Storage{std::vector<std::byte>data;std::size_t origin;};
Storage allocate(const Shape&shape,const Strides&strides,std::size_t bytes){
 std::ptrdiff_t low=0,high=0;
 for(std::size_t i=0;i<shape.size();++i){auto last=std::ptrdiff_t(shape[i]?shape[i]-1:0)*strides[i];low+=std::min(last,std::ptrdiff_t(0));high+=std::max(last,std::ptrdiff_t(0));}
 return{std::vector<std::byte>(std::size_t(high-low+1)*bytes+256,std::byte{0x55}),128+std::size_t(-low)*bytes};
}
Strides layout(const Shape&shape,std::mt19937&gen,int mode){
 Strides s(shape.size());std::vector<std::size_t>order(shape.size());std::iota(order.begin(),order.end(),0);
 if(mode==1)std::reverse(order.begin(),order.end());else if(mode>=2)std::shuffle(order.begin(),order.end(),gen);
 std::ptrdiff_t stride=1;
 for(auto it=order.rbegin();it!=order.rend();++it){s[*it]=stride;stride*=std::ptrdiff_t(std::max(std::size_t(1),shape[*it]));if(mode>=2&&gen()%3==0)stride+=1+gen()%5;}
 if(mode==3)for(auto&v:s)if(gen()%4==0)v=-v;
 if(mode==4)for(auto&v:s)if(gen()%3==0)v=0;
 return s;
}
int main(){
 std::mt19937 gen(2515);std::uint64_t cases=0,blocks=0,leaves=0;
 for(std::size_t bytes:{1u,2u,4u,8u})for(int n=0;n<2400;++n){
  Shape shape(std::size_t(n%6));for(auto&v:shape)v=gen()%8;
  auto ss=layout(shape,gen,n%5),ds=layout(shape,gen,(n/5)%5);auto source=allocate(shape,ss,bytes),out=allocate(shape,ds,bytes);auto expected=out.data;
  for(auto&v:source.data)v=std::byte(gen()%256);auto before=source.data;
  original(expected.data()+out.origin,source.data.data()+source.origin,shape,ds,ss,bytes);
  infinilm::runtime::detail::CpuCopyStatsR2515 stats;
  infinilm::runtime::detail::copy_cpu_strided_r2515(out.data.data()+out.origin,source.data.data()+source.origin,shape,ds,ss,bytes,0,stats);
  require(out.data==expected,"full destination incl padding/guards differs from original");require(source.data==before,"source changed");blocks+=stats.blocks;leaves+=stats.scalar_leaves;++cases;
 }
 // Model CPU copies gather first, then scatter: explicitly retain overlap order.
 for(std::size_t bytes:{1u,2u,4u})for(bool reverse:{false,true}){
  Shape shape{9,13};Strides ss{17,reverse?-1:1},ds{15,1};auto storage=allocate(shape,ss,bytes);storage.data.resize(storage.data.size()+5000,std::byte{0x55});for(auto&v:storage.data)v=std::byte(gen()%256);auto reference=storage.data,actual=storage.data;std::vector<std::byte>tmp(9*13*bytes),tmp2(tmp.size());Strides contiguous{13,1};
  original(tmp.data(),reference.data()+storage.origin,shape,contiguous,ss,bytes);original(reference.data()+storage.origin+3*bytes,tmp.data(),shape,ds,contiguous,bytes);
  infinilm::runtime::detail::CpuCopyStatsR2515 a,b;infinilm::runtime::detail::copy_cpu_strided_r2515(tmp2.data(),actual.data()+storage.origin,shape,contiguous,ss,bytes,0,a);infinilm::runtime::detail::copy_cpu_strided_r2515(actual.data()+storage.origin+3*bytes,tmp2.data(),shape,ds,contiguous,bytes,0,b);require(actual==reference,"staged overlapping views differ");++cases;
 }
 std::cout<<"{\"completed\":true,\"full_storage_byte_cases\":"<<cases<<",\"bulk_blocks\":"<<blocks<<",\"scalar_leaves\":"<<leaves<<",\"negative_zero_stride_singleton_empty_scalar_overlap_covered\":true}"<<std::endl;
}
