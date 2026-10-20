#pragma once
#include "deepseek_v4_ffn_routed_block_r20.hpp"
#include "deepseek_v4_swa_attention.hpp"
#include "deepseek_v4_csa_attention.hpp"
#include "deepseek_v4_hca_attention.hpp"

namespace infinilm::models::deepseek_v4 {
// Owns all state and checkpoint bindings of one actual V4 decoder layer.
// Parent traversal sees flat aliases; this class finalizes its owned modules.
class NativeDecoderR20 final:public weights::ModelWeights {
public:
 using Request=TemporalCompressor::Request;
 struct Output {
  runtime::Tensor hidden,normalized_attention,attention_value,attention_residual;
  HyperConnection::PreResult attention_prepared;
  DeepseekV4FFNBlockR20::DetailedOutput ffn;
 };
 NativeDecoderR20(runtime::Runtime&,const nlohmann::json&,size_t layer,
               const engine::distributed::RankInfo&,size_t rows);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Output forward(runtime::Runtime&,const runtime::Tensor&residual,
                const runtime::Tensor&token_ids,const runtime::Tensor&positions,
                const std::vector<Request>&);
 void reset(runtime::Runtime&);
 std::vector<runtime::Tensor>cache_tensors()const;
private:
 void check_owner(runtime::Runtime&)const;
 runtime::Runtime*owner_;size_t rows_,maximum_,vocab_;int ratio_;
 bool ready_=false,poisoned_=false;
 std::vector<size_t>lengths_;
 std::unique_ptr<HyperConnection>hc_;
 std::unique_ptr<layers::RMSNorm>norm_;
 std::unique_ptr<DeepseekV4FFNBlockR20>ffn_;
 std::unique_ptr<SwaAttention>swa_;
 std::unique_ptr<CsaAttention>csa_;
 std::unique_ptr<HcaAttention>hca_;
};
}
