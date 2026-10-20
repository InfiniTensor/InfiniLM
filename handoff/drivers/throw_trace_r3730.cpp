#include <dlfcn.h>
#include <execinfo.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <typeinfo>
#include <unistd.h>
extern "C" __attribute__((noreturn)) void __cxa_throw(void*object,std::type_info*type,void(*destroy)(void*)){
 using Fn=void(*)(void*,std::type_info*,void(*)(void*));static Fn real=reinterpret_cast<Fn>(dlsym(RTLD_NEXT,"__cxa_throw"));static thread_local bool printing=false;
 if(!printing&&type&&std::strcmp(type->name(),typeid(std::invalid_argument).name())==0){
  const char*message=static_cast<std::invalid_argument*>(object)->what();
  if(message&&std::strstr(message,"operator has no active implementation")){
   printing=true;std::fprintf(stderr,"OP_ACTIVE_IMPL_THROW_R3730 pid=%d type=%s message=%s\n",int(getpid()),type->name(),message);void*frames[48];int n=backtrace(frames,48);backtrace_symbols_fd(frames,n,STDERR_FILENO);std::fflush(stderr);printing=false;
  }
 }
 if(!real){std::fprintf(stderr,"R3730 missing real __cxa_throw\n");_exit(127);}real(object,type,destroy);__builtin_unreachable();
}
