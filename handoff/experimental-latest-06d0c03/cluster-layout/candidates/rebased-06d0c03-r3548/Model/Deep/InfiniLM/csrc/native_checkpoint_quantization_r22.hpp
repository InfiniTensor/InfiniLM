#pragma once
#include "layers/quantization/base_quantization.hpp"
namespace infinilm::quantization {
// The model binds its actual checkpoint INT8/FP8 tensors and calls public Core
// operators. This descriptor must never fall through to generic BF16 Linear.
class NativeCheckpointQuantization final:public BaseQuantization {
public:
 explicit NativeCheckpointQuantization(const nlohmann::json&config):BaseQuantization(config){}
 QuantScheme get_quant_scheme()const override{return QuantScheme::MODEL_NATIVE;}
 std::vector<ParamDescriptor>get_param_layout(size_t,size_t,int,int,int,int,const infinilm::runtime::DataType&,bool)const override{throw std::logic_error("Native checkpoint layout is owned by its typed model");}
 infinilm::runtime::Tensor forward(infinilm::runtime::Runtime&,const ParamsMap&,const infinilm::runtime::Tensor&,bool,float)const override{throw std::logic_error("Native checkpoint requires its typed public-Core model execution");}
 std::vector<SplitParam>split_params(const std::unordered_map<std::string,infinilm::weights::Weight>&,const std::vector<SplitInfo>&,int,int,int,int)const override{throw std::logic_error("Native checkpoint shards are bound by its typed model");}
};
}
