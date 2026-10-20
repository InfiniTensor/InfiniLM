#include "glm_fp8_capcache_r3806.hpp"
#include <algorithm>
#include <array>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
extern "C" const char*glm_fp8_error_r3806();
extern "C" void*glm_fp8_create_r3806(int,int,size_t,size_t,size_t,size_t,size_t,size_t,size_t,void*);
extern "C" int glm_fp8_run_m_r3806(void*,size_t,void*,const void*,const void*,const void*,const void*);
extern "C" int glm_fp8_destroy_r3806(void*);
namespace {
thread_local std::string error;constexpr size_t limit=768ull*1024*1024,entries_limit=16;
using Key=std::array<size_t,7>;
struct Geometry{Key key;size_t bytes;};
Geometry geometry(size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro,size_t co){
 if(m<32||m>4096||!n||n>32768||!k||k>32768||!sr||sr>512||!sc||sc>512||ro>sr*128||n>sr*128-ro||co>sc*128||k>sc*128-co)throw std::invalid_argument("3806 bounded prefill geometry");
 size_t cap=32;while(cap<m)cap*=2;const size_t blocks=(co+k+127)/128-co/128;const size_t bytes=blocks*(2*cap*128+2*n*128+4*std::min(cap,size_t(256))*n);if(bytes>limit)throw std::invalid_argument("3806 one plan exceeds768MiB");return{{cap,n,k,sr,sc,ro,co},bytes};
}
struct Entry{void*plan;size_t bytes;uint64_t last;};
struct Cache{int device;void*stream;const void*identity;std::thread::id owner;std::map<Key,Entry>entries;uint64_t created{},destroyed{},hits{},misses{},evictions{},bytes{},highwater{},runs{},invalid{},max_m{},max_cap{},clock{};};
void owned(Cache*c,const void*identity,void*stream){if(!c||c->owner!=std::this_thread::get_id()||c->identity!=identity||c->stream!=stream)throw std::invalid_argument("3806 Runtime/stream/owner mismatch");}
void remove(Cache&c,std::map<Key,Entry>::iterator i){if(glm_fp8_destroy_r3806(i->second.plan))std::terminate();c.bytes-=i->second.bytes;++c.destroyed;c.entries.erase(i);}
}
extern "C" const char*glm_fp8_capcache_error_r3806(){return error.c_str();}
extern "C" int glm_fp8_capcache_fits_r3806(size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro,size_t co){try{geometry(m,n,k,sr,sc,ro,co);return 1;}catch(...){return 0;}}
extern "C" void*glm_fp8_capcache_create_r3806(int device,void*stream,const void*identity){try{if(device<0||device>=8||!stream||!identity)throw std::invalid_argument("3806 missing owner resource");auto*c=new Cache;c->device=device;c->stream=stream;c->identity=identity;c->owner=std::this_thread::get_id();return c;}catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int glm_fp8_capcache_run_r3806(void*raw,const void*identity,void*stream,size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro,size_t co,void*out,const void*x,const void*xs,const void*w,const void*ws){auto*c=static_cast<Cache*>(raw);try{
 owned(c,identity,stream);if(!out||!x||!xs||!w||!ws)throw std::invalid_argument("3806 missing data");const auto g=geometry(m,n,k,sr,sc,ro,co);auto i=c->entries.find(g.key);
 if(i==c->entries.end()){
  while(c->entries.size()>=entries_limit||c->bytes+g.bytes>limit){auto victim=std::min_element(c->entries.begin(),c->entries.end(),[](const auto&a,const auto&b){return a.second.last<b.second.last;});if(victim==c->entries.end())throw std::logic_error("3806 impossible cache capacity");remove(*c,victim);++c->evictions;}
  void*p=glm_fp8_create_r3806(c->device,1,g.key[0],n,k,sr,sc,ro,co,stream);if(!p)throw std::runtime_error(glm_fp8_error_r3806());try{i=c->entries.emplace(g.key,Entry{p,g.bytes,++c->clock}).first;}catch(...){if(glm_fp8_destroy_r3806(p))std::terminate();throw;}++c->created;++c->misses;c->bytes+=g.bytes;c->highwater=std::max(c->highwater,c->bytes);
 }else{++c->hits;i->second.last=++c->clock;}
 if(glm_fp8_run_m_r3806(i->second.plan,m,out,x,xs,w,ws))throw std::runtime_error(glm_fp8_error_r3806());++c->runs;c->max_m=std::max(c->max_m,uint64_t(m));c->max_cap=std::max(c->max_cap,uint64_t(g.key[0]));return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->owner==std::this_thread::get_id())++c->invalid;return 1;}}
extern "C" int glm_fp8_capcache_stats_r3806(void*raw,uint64_t*out,size_t count){try{auto*c=static_cast<Cache*>(raw);if(!c||c->owner!=std::this_thread::get_id()||!out||count!=12)throw std::invalid_argument("3806 stats owner/bounds");std::array<uint64_t,12>v={c->created,c->destroyed,c->hits,c->misses,c->evictions,c->entries.size(),c->bytes,c->highwater,c->runs,c->invalid,c->max_m,c->max_cap};std::copy(v.begin(),v.end(),out);return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_fp8_capcache_clear_r3806(void*raw,const void*identity,void*stream){try{auto*c=static_cast<Cache*>(raw);owned(c,identity,stream);while(!c->entries.empty())remove(*c,c->entries.begin());return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_fp8_capcache_destroy_r3806(void*raw,const void*identity,void*stream){try{auto*c=static_cast<Cache*>(raw);owned(c,identity,stream);while(!c->entries.empty())remove(*c,c->entries.begin());delete c;return 0;}catch(const std::exception&e){error=e.what();return 1;}}
