#include <infini/ops.h>
#include <nlohmann/json.hpp>
#include <iostream>
int main(){
 using namespace infini::ops;auto device=infini::rt::Device::Type::kHygon;
 nlohmann::json d;
 d["IndexSelect"]=IndexSelect::active_implementation_indices(device);
 d["Gemm"]=Gemm::active_implementation_indices(device);
 d["RmsNorm"]=RmsNorm::active_implementation_indices(device);
 d["IndexedKimiDeltaAttention"]=IndexedKimiDeltaAttention::active_implementation_indices(device);
 std::cout<<d.dump()<<std::endl;
}
