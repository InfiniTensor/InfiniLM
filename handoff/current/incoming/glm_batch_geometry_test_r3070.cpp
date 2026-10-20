#include "glm_batch_cache_geometry_r3068.hpp"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
int main(){
 using glm_batch_cache_r3068::fits;std::size_t cases=0;
 struct Shape{std::size_t n,k,ro,co,sc;};
 const Shape shapes[]={{2048,6144,0,0,48},{2048,2048,14336,0,16},
 {576,6144,0,0,48},{3584,512,25088,0,4},{6144,2048,0,14336,128}};
 for(auto s:shapes)for(auto m:{32u,64u}){assert(fits(m,s.n,s.k,(s.ro+s.n+127)/128,s.sc,s.ro,s.co));++cases;}
 for(auto m:{0u,1u,8u,16u,31u,33u,63u,65u,128u,8192u}){assert(!fits(m,576,6144,5,48,0,0));++cases;}
 for(auto m:{32u,64u}){
  assert(!fits(m,32768,32768,256,256,0,0));++cases;
  assert(!fits(m,576,6144,4,48,0,0));++cases;
  assert(!fits(m,576,6144,5,47,0,0));++cases;
  assert(!fits(m,1,1,1,1,128,0));++cases;
  assert(!fits(m,1,1,1,1,0,128));++cases;
  assert(!fits(m,0,1,1,1,0,0));++cases;
  assert(!fits(m,1,0,1,1,0,0));++cases;
  assert(!fits(m,1,1,513,1,0,0));++cases;
  assert(!fits(m,1,1,1,513,0,0));++cases;
  const auto z=std::numeric_limits<std::size_t>::max();
  for(int which=0;which<6;++which){std::size_t a[]={1,1,1,1,0,0};a[which]=z;assert(!fits(m,a[0],a[1],a[2],a[3],a[4],a[5]));++cases;}
 }
 std::cout<<"{\"passed\":true,\"cases\":"<<cases<<",\"scope\":\"CPU admission/fallback/overflow only, not model proof\"}\n";
}
