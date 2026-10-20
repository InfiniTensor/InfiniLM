#include <infini/ops.h>
#include <nlohmann/json.hpp>
#include <iostream>
int main(){using namespace infini::ops;nlohmann::json d;for(auto v:{infini::rt::Device::Type::kHygon,infini::rt::Device::Type::kCpu}){
 auto key=v==infini::rt::Device::Type::kHygon?"Hygon":"CPU";d[key]["Embedding"]=Embedding::active_implementation_indices(v);d[key]["IndexSelect"]=IndexSelect::active_implementation_indices(v);try{auto c=Embedding::DefaultConfig(v);d[key]["Embedding_default_config_ok"]=true;}catch(const std::exception&e){d[key]["Embedding_default_error"]=e.what();}}
 std::cout<<d.dump()<<std::endl;
}
