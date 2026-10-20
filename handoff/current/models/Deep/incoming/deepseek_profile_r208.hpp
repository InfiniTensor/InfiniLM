#pragma once
#include <chrono>
extern "C" void modern_deepseek_profile_r208_record(int,int,int,double);
namespace deep_profile_r208 {
inline thread_local int mode=0;
struct Stamp {
    int device,phase;
    std::chrono::steady_clock::time_point last=std::chrono::steady_clock::now();
    Stamp(int d,int p):device(d),phase(p){}
    template<class Runtime>void mark(Runtime&r,int component){
        r.syncStream();auto now=std::chrono::steady_clock::now();
        modern_deepseek_profile_r208_record(device,phase,component,std::chrono::duration<double>(now-last).count());last=now;
    }
    void skip(){last=std::chrono::steady_clock::now();}
};
}
