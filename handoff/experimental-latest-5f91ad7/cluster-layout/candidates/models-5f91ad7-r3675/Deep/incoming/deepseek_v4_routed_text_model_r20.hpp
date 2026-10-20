#pragma once
#include "deepseek_v4_native_routed_decoder_r20.hpp"

namespace infinilm::models::deepseek_v4 {
class TextHeadR20 final:public weights::ModelWeights {
public:
 struct Output{runtime::Tensor collapsed,normalized,logits,pre;};
 TextHeadR20(runtime::Runtime&,const nlohmann::json&,const engine::distributed::RankInfo&);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&)const;
private:
 runtime::Runtime*owner_;engine::distributed::RankInfo rank_;size_t vocab_;bool ready_=false;
 runtime::Tensor fn_,base_,scale_,head_,constants_;
 std::unique_ptr<layers::RMSNorm>norm_;
};

// Actual43-layer main causal model. The optional separate MTP predictor is not
// invoked by ordinary autoregressive decoding. TP1/8, PP1 only at this stage.
class TextModelR20 final:public weights::ModelWeights {
public:
 using Request=NativeDecoderR20::Request;
 struct Output{runtime::Tensor residual;TextHeadR20::Output head;};
 TextModelR20(runtime::Runtime&,const nlohmann::json&,const engine::distributed::RankInfo&,size_t rows,size_t capacity);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<Request>&,bool all_positions=false);
 void reset(runtime::Runtime&);
private:
 runtime::Runtime*owner_;size_t rows_,capacity_,vocab_;bool ready_=false,poisoned_=false;
 runtime::Tensor embedding_;
 std::vector<size_t>lengths_;
 std::vector<std::unique_ptr<NativeDecoderR20>>layers_;
 std::unique_ptr<TextHeadR20>head_;
};
}
