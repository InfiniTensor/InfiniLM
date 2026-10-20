#pragma once
#include "deepseek_v4_temporal_compressor.hpp"
#include "engine/distributed/communication_group.hpp"
namespace infinilm::models::deepseek_v4 {
class PagedIndexerCache {
public:
 struct Packed{runtime::Tensor values,scales;};
 PagedIndexerCache(runtime::Runtime&,size_t pages);
 void store(runtime::Runtime&,const std::vector<int64_t>&slots,const Packed&);
 Packed gather(runtime::Runtime&,const std::vector<int64_t>&slots)const;
 void reset(runtime::Runtime&);
 const runtime::Tensor&raw()const{return raw_;}
private:
 void validate(runtime::Runtime&,const std::vector<int64_t>&slots,bool write)const;
 std::shared_ptr<runtime::Runtime>owner_;size_t pages_;runtime::Tensor raw_;std::vector<uint8_t>written_;
};
class C4Indexer final:public weights::ModelWeights {
public:
 using Request=TemporalCompressor::Request;
 struct Pending{
  runtime::Tensor query,query_scales,merged_weights,raw_query,head_weights,projected_score,pooled;
  PagedIndexerCache::Packed new_keys;
  std::vector<runtime::Tensor>scores,indices;
  std::vector<int64_t>slots;
 private:
  friend class C4Indexer;
  TemporalCompressor::Batch compression;
  std::vector<Request>requests;
  const C4Indexer*owner=nullptr;size_t revision=0;
 };
 C4Indexer(runtime::Runtime&,const nlohmann::json&,size_t layer,const engine::distributed::RankInfo&,size_t rows,size_t capacity,size_t topk=512);
 void process_weights_after_loading(runtime::Runtime&,bool)override;
 Pending prepare(runtime::Runtime&,const runtime::Tensor&hidden,const runtime::Tensor&q_lora,const runtime::Tensor&positions,const std::vector<Request>&)const;
 void commit(runtime::Runtime&,Pending&);
 Pending forward(runtime::Runtime&,const runtime::Tensor&,const runtime::Tensor&,const runtime::Tensor&,const std::vector<Request>&);
 void reset(runtime::Runtime&);
 const runtime::Tensor&raw_cache()const{return cache_->raw();}
 const runtime::Tensor&state()const{return compressor_->state();}
 size_t local_heads()const{return heads_;}
private:
 void check_owner(runtime::Runtime&)const;
 runtime::Tensor rope(runtime::Runtime&,const std::vector<int64_t>&)const;
 runtime::Runtime*owner_;size_t rows_,capacity_,topk_,heads_,maximum_,original_,revision_=0;
 double theta_,factor_,beta_fast_,beta_slow_;float eps_;int tp_;infinicclComm_t comm_;bool ready_=false,poisoned_=false;
 runtime::Tensor query_weight_,query_scale_,head_weight_,norm_weight_,score_scale_;
 std::unique_ptr<TemporalCompressor>compressor_;std::unique_ptr<PagedIndexerCache>cache_;std::vector<size_t>lengths_;
};
}
