#if defined(REGISTRY_FIRST_R3717)
#include <infini/ops.h>
#endif
#include "kimi_spec_math_r3676.hpp"
#include <infini/ops.h>
#include <nlohmann/json.hpp>
#include <iostream>
int main(){using namespace infini::ops;auto h=infini::rt::Device::Type::kHygon,c=infini::rt::Device::Type::kCpu;nlohmann::json d;
#define CHECK(OP) d[#OP]={{"Hygon",OP::active_implementation_indices(h)},{"CPU",OP::active_implementation_indices(c)}};
 CHECK(IndexSelect) CHECK(Gemm) CHECK(RmsNorm) CHECK(IndexedKimiDeltaAttention) CHECK(Cast) CHECK(Add) CHECK(Mul) CHECK(Softmax)
 std::cout<<d.dump()<<std::endl;
}
