#include "glm_fp8_cache_r807.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>

extern "C" const char*glm_fp8_error_r776();
extern "C" void*glm_fp8_create_r776(int,int,size_t,size_t,size_t,size_t,size_t,size_t,size_t,void*);
extern "C" int glm_fp8_run_r776(void*,int,void*,const void*,const void*,const void*,const void*);
extern "C" int glm_fp8_destroy_r776(void*);
namespace {
thread_local std::string error;
constexpr size_t byte_limit=768ull*1024*1024,entry_limit=20;
using Key=std::array<size_t,8>; // dtype,M,N,K,scaleRows,scaleCols,rowOffset,colOffset
struct Entry {void*plan{};size_t bytes{};uint64_t last{};};
struct Cache {
 int device{};void*stream{};const void*identity{};std::thread::id owner;
 std::map<Key,Entry>entries;
 uint64_t created{},destroyed{},hits{},misses{},evictions{},bytes{},highwater{},runs{},invalid{},clock{};
};
void owned(const Cache*c,const void*identity,void*stream){
 if(!c||c->owner!=std::this_thread::get_id())throw std::invalid_argument("807 owner thread differs");
 if(c->identity!=identity||c->stream!=stream)throw std::invalid_argument("807 Runtime identity or stream differs");
}
size_t geometry(const Key&k){
 const auto[dtype,m,n,inner,sr,sc,ro,co]=k;
 if(dtype!=1||(m!=8&&m!=16&&m!=32&&m!=64&&m!=256&&m!=512&&m!=1024&&m!=2048&&m!=4096)||!n||n>32768||!inner||inner>32768||!sr||sr>512||!sc||sc>512||ro>sr*128||n>sr*128-ro||co>sc*128||inner>sc*128-co)throw std::invalid_argument("3124 bounded BF16 decode/prefill geometry invalid");
 size_t blocks=(co+inner+127)/128-co/128;
 size_t bytes=blocks*(2*m*128+2*n*128+4*std::min(m,size_t(256))*n);
 if(bytes>192ull*1024*1024)throw std::invalid_argument("3124 entry workspace exceeds192MiB");
 return bytes;
}
void remove(Cache&c,std::map<Key,Entry>::iterator i){
 // The original helper drains the recorded stream under its device guard.
 if(glm_fp8_destroy_r776(i->second.plan))std::terminate();
 c.bytes-=i->second.bytes;++c.destroyed;c.entries.erase(i);
}
}
extern "C" const char*glm_fp8_cache_error_r807(){return error.c_str();}
extern "C" void*glm_fp8_cache_create_r807(int device,void*stream,const void*identity){try{
 if(device<0||device>=8||!identity)throw std::invalid_argument("807 missing Runtime identity or invalid device");
 auto*c=new Cache;c->device=device;c->stream=stream;c->identity=identity;c->owner=std::this_thread::get_id();return c;
 }catch(const std::exception&e){error=e.what();return nullptr;}}
extern "C" int glm_fp8_cache_run_r807(void*raw,const void*identity,void*stream,int mode,int dtype,size_t m,size_t n,size_t k,size_t sr,size_t sc,size_t ro,size_t co,void*out,const void*x,const void*xs,const void*w,const void*ws){
 auto*c=static_cast<Cache*>(raw);try{
 owned(c,identity,stream);
 if(mode<0||mode>1||!out||!x||!xs||!w||!ws)throw std::invalid_argument("807 mode or pointer invalid");
 Key key{size_t(dtype),m,n,k,sr,sc,ro,co};size_t required=geometry(key);
 auto i=c->entries.find(key);
 if(i==c->entries.end()){
  while(c->entries.size()>=entry_limit||c->bytes+required>byte_limit){
   auto victim=std::min_element(c->entries.begin(),c->entries.end(),[](const auto&a,const auto&b){return a.second.last<b.second.last;});
   if(victim==c->entries.end())throw std::logic_error("807 impossible bounded cache state");
   remove(*c,victim);++c->evictions;
  }
  void*p=glm_fp8_create_r776(c->device,dtype,m,n,k,sr,sc,ro,co,c->stream);
  if(!p)throw std::runtime_error(glm_fp8_error_r776());
  try{i=c->entries.emplace(key,Entry{p,required,++c->clock}).first;}catch(...){if(glm_fp8_destroy_r776(p))std::terminate();throw;}
  ++c->created;++c->misses;c->bytes+=required;c->highwater=std::max(c->highwater,c->bytes);
 }else{++c->hits;i->second.last=++c->clock;}
 // Every invocation supplies and unpacks the current X/W/scales. Only scratch
 // and a deterministic BLAS handle are cached; no decoded weights survive.
 if(glm_fp8_run_r776(i->second.plan,mode,out,x,xs,w,ws))throw std::runtime_error(glm_fp8_error_r776());
 ++c->runs;return 0;
 }catch(const std::exception&e){error=e.what();if(c&&c->owner==std::this_thread::get_id())++c->invalid;return 1;}}
extern "C" int glm_fp8_cache_stats_r807(void*raw,uint64_t*out,size_t count){try{
 auto*c=static_cast<Cache*>(raw);if(!c||c->owner!=std::this_thread::get_id()||!out||count!=10)throw std::invalid_argument("807 stats owner/shape invalid");
 std::array<uint64_t,10>v{c->created,c->destroyed,c->hits,c->misses,c->evictions,c->entries.size(),c->bytes,c->highwater,c->runs,c->invalid};std::copy(v.begin(),v.end(),out);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_fp8_cache_clear_r807(void*raw,const void*identity,void*stream){try{
 auto*c=static_cast<Cache*>(raw);owned(c,identity,stream);while(!c->entries.empty())remove(*c,c->entries.begin());return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int glm_fp8_cache_destroy_r807(void*raw,const void*identity,void*stream){try{
 auto*c=static_cast<Cache*>(raw);owned(c,identity,stream);while(!c->entries.empty())remove(*c,c->entries.begin());delete c;return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
