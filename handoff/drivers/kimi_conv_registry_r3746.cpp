#include "kimi_spec_math_r3676.hpp"
#include "execution/submit.hpp"
#include "base/causal_conv1d.h"
#include <nlohmann/json.hpp>
#include <iostream>
int main(){using C=infini::ops::CausalConv1d;auto device=infini::rt::Device::Type::kHygon;nlohmann::json d;d["active"]=C::active_implementation_indices(device);try{auto c=C::DefaultConfig(device);d["default_config_ok"]=true;d["implementation"]=c.implementation_index();}catch(const std::exception&e){d["default_config_ok"]=false;d["error"]=e.what();}std::cout<<d.dump()<<std::endl;}
