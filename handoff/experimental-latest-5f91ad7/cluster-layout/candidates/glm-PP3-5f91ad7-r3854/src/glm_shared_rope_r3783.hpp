#pragma once
#include "layers/rotary_embedding/rope.hpp"
#include "runtime/runtime.hpp"
#include <map>
#include <tuple>
#include <array>
#include <cstdint>
namespace infinilm::models::glm_moe_dsa::shared_rope_r3783 {
namespace rt=infinilm::runtime;
using Key=std::tuple<uintptr_t,size_t,size_t,size_t,double>;
inline thread_local std::map<Key,std::weak_ptr<layers::RoPE>>pool;
inline thread_local std::map<rt::Runtime*,std::array<uint64_t,3>>counters;
inline thread_local std::map<rt::Runtime*,std::weak_ptr<rt::Runtime>>owner_tokens;
inline std::shared_ptr<layers::RoPE>get(rt::Runtime&r,size_t head,size_t rotary,size_t maximum,double theta){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||rotary!=64||(head!=64&&head!=128)||!maximum||theta!=8000000.)throw std::invalid_argument("3783 qualified GLM default RoPE geometry");
 if(owner_tokens[&r].expired()){for(auto it=pool.begin();it!=pool.end();)if(std::get<0>(it->first)==reinterpret_cast<uintptr_t>(&r))it=pool.erase(it);else ++it;counters.erase(&r);owner_tokens[&r]=r.shared_from_this();}
 Key key{reinterpret_cast<uintptr_t>(&r),head,rotary,maximum,theta};auto&count=counters[&r];++count[0];auto&entry=pool[key];if(auto p=entry.lock()){++count[2];return p;}
 auto result=std::make_shared<layers::RoPE>(r,head,rotary,maximum,theta,layers::RoPE::Algo::GPT_J,rt::DataType::kBFloat16,r.device());entry=result;++count[1];return result;
}
inline std::array<uint64_t,5>stats(rt::Runtime&r){r.assertOwnerThread();const auto c=counters[&r];uint64_t keys=0,live=0;for(const auto&v:pool)if(std::get<0>(v.first)==reinterpret_cast<uintptr_t>(&r)){++keys;live+=!v.second.expired();}return{c[0],c[1],c[2],keys,live};}
}
