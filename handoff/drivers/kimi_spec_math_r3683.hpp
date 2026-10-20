#pragma once
#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"
#include <array>
#include <atomic>
#include <cstdlib>
#include <string>
#include <stdexcept>
namespace infinilm::models::kimi_native {
namespace rt=infinilm::runtime;
struct MathFrameR3676 {rt::Runtime*owner;int mode;};
inline thread_local const MathFrameR3676*math_frame_r3676=nullptr;
inline std::array<std::array<std::atomic<uint64_t>,3>,8>math_counts_r3676{};
class SpecMathScopeR3676 final {
 MathFrameR3676 frame_;
public:
 SpecMathScopeR3676(rt::Runtime&r,bool verification):frame_{&r,0}{
  r.assertOwnerThread();
  if(!verification)return;
  const char*p=std::getenv("INFINILM_KIMI_SPEC_MATH_R3676");
  if(!p||std::string(p)=="0")return;
  if(std::string(p)=="1")frame_.mode=1;else if(std::string(p)=="2")frame_.mode=2;else throw std::invalid_argument("3676 verification math policy must0/1/2");
  if(math_frame_r3676||r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||r.device().index()<0||r.device().index()>=8)throw std::invalid_argument("3676 verification math requires dedicated eagerHygon owner");
  math_frame_r3676=&frame_;
 }
 ~SpecMathScopeR3676()noexcept{if(frame_.mode){if(math_frame_r3676!=&frame_)std::terminate();math_frame_r3676=nullptr;}}
 SpecMathScopeR3676(const SpecMathScopeR3676&)=delete;
 SpecMathScopeR3676&operator=(const SpecMathScopeR3676&)=delete;
};
inline int spec_math_mode_r3676(rt::Runtime&r){r.assertOwnerThread();if(!math_frame_r3676)return 0;if(math_frame_r3676->owner!=&r)throw std::logic_error("3676 different Runtime owner");return math_frame_r3676->mode;}
inline bool spec_decode_projection_r3676(rt::Runtime&r){if(spec_math_mode_r3676(r)!=2)return false;++math_counts_r3676.at(r.device().index())[2];return true;}
void spec_kda_roundstep_r3676(rt::Runtime&,rt::Tensor out,rt::Tensor state,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&v,const rt::Tensor&gate,const rt::Tensor&beta,const rt::Tensor&alog,const rt::Tensor&bias,const rt::Tensor&offsets,const rt::Tensor&read,const rt::Tensor&write,float scale,float lower,bool norm);
}
