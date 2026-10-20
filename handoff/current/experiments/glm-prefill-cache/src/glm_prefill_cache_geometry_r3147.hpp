#pragma once
#include <cstddef>
namespace glm_prefill_cache_r3147 {
inline bool fits(std::size_t m,std::size_t n,std::size_t k,std::size_t sr,std::size_t sc,std::size_t ro,std::size_t co){
 if((m!=256&&m!=512&&m!=1024&&m!=2048&&m!=4096)||!n||n>32768||!k||k>32768||!sr||sr>512||!sc||sc>512||ro>sr*128||n>sr*128-ro||co>sc*128||k>sc*128-co)return false;
 return ((co+k+127)/128-co/128)*(2*m*128+2*n*128+4*256*n)<=192ull*1024*1024;
}
}
