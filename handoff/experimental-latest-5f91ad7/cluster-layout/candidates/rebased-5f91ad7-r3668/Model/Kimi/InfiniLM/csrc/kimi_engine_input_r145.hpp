#pragma once
#include "cache/kv_cache.hpp"
#include "runtime/tensor.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>
namespace infinilm::models::kimi_native {
namespace rt=infinilm::runtime;
struct Request{size_t row,past,tokens;};
inline void validate_cache(const cache::CacheConfig*base){
 const auto*c=dynamic_cast<const cache::PagedKVCacheConfig*>(base);
 if(!c||c->block_size()!=256||!((c->max_batch_size()>=1&&c->max_batch_size()<=128)&&c->num_blocks()==4*c->max_batch_size()+4))throw std::invalid_argument("Kimi Engine requires rows1..128, pages4rows+4 at256 width");
}
inline size_t request_capacity(const cache::CacheConfig*base){validate_cache(base);return dynamic_cast<const cache::PagedKVCacheConfig*>(base)->max_batch_size();}
inline rt::Tensor freeze_cpu(const std::optional<rt::Tensor>&source,rt::DataType dtype,size_t ndim){
 if(!source||!source.value()||source.value()->device().type()!=rt::Device::Type::kCpu||source.value()->dtype()!=dtype||source.value()->ndim()!=ndim||!source.value()->is_contiguous())throw std::invalid_argument("Kimi Engine requires original contiguous typed CPU metadata");
 const auto&t=source.value();auto storage=std::make_shared<std::vector<std::byte>>(t->nbytes());if(t->nbytes())std::memcpy(storage->data(),t->data(),t->nbytes());return rt::Tensor::from_blob(storage->data(),t->shape(),dtype,{rt::Device::Type::kCpu,0},storage);
}
template<class T>std::vector<T>values(const rt::Tensor&t){if(!t->numel())return {};const auto*p=reinterpret_cast<const T*>(t->data());return {p,p+t->numel()};}
template<class Input>std::vector<Request>freeze_input(Input&a,const std::vector<size_t>&lengths,size_t vocab){
 if((lengths.size()<2||lengths.size()>129)||a.sample_all_positions||a.nsa_owner_snapshot||a.pixel_values||a.image_bound||a.tgt_sizes||a.image_grid_thw||a.image_req_ids||a.visual_token_ranges||a.target_hidden_states||a.random_values||a.sampling_rows||a.top_k!=1||a.top_p!=1.f||a.temperature!=1.f)throw std::invalid_argument("Kimi Engine initial slice is eager text greedy last-token sampling only");
 a.input_ids=freeze_cpu(a.input_ids,rt::DataType::kInt64,2);a.position_ids=freeze_cpu(a.position_ids,rt::DataType::kInt64,2);
 a.past_sequence_lengths=freeze_cpu(a.past_sequence_lengths,rt::DataType::kInt32,1);a.total_sequence_lengths=freeze_cpu(a.total_sequence_lengths,rt::DataType::kInt32,1);a.input_offsets=freeze_cpu(a.input_offsets,rt::DataType::kInt32,1);a.cu_seqlens=freeze_cpu(a.cu_seqlens,rt::DataType::kInt32,1);
 a.mamba_init_state_indices=freeze_cpu(a.mamba_init_state_indices,rt::DataType::kInt32,1);a.mamba_final_state_indices=freeze_cpu(a.mamba_final_state_indices,rt::DataType::kInt32,1);
 a.block_tables=freeze_cpu(a.block_tables,rt::DataType::kInt32,2);a.slot_mapping=freeze_cpu(a.slot_mapping,rt::DataType::kInt64,1);
 const auto ids=values<int64_t>(a.input_ids.value()),positions=values<int64_t>(a.position_ids.value()),slots=values<int64_t>(a.slot_mapping.value());
 const auto past=values<int32_t>(a.past_sequence_lengths.value()),total=values<int32_t>(a.total_sequence_lengths.value()),offsets=values<int32_t>(a.input_offsets.value()),cu=values<int32_t>(a.cu_seqlens.value()),rows=values<int32_t>(a.mamba_init_state_indices.value()),writes=values<int32_t>(a.mamba_final_state_indices.value()),table=values<int32_t>(a.block_tables.value());
 size_t n=rows.size(),tokens=ids.size();if(!n||n>=lengths.size()||rows!=writes||!tokens||tokens>8192||a.input_ids.value()->shape()!=rt::Shape{1,tokens}||a.position_ids.value()->shape()!=rt::Shape{1,tokens}||past.size()!=n||total.size()!=n||offsets.size()!=n+1||cu.size()!=n+1||offsets[0]!=0||cu[0]!=0||offsets.back()<0||size_t(offsets.back())!=tokens||slots.size()!=tokens||a.block_tables.value()->shape()!=rt::Shape{n,4})throw std::invalid_argument("Kimi packed CPU request geometry differs");
 for(auto id:ids)if(id<0||size_t(id)>=vocab)throw std::invalid_argument("Kimi token outside vocabulary");
 std::set<size_t>seen;std::vector<Request>requests;
 for(size_t i=0;i<n;++i){
  if(rows[i]<1||size_t(rows[i])>=lengths.size()||!seen.insert(size_t(rows[i])).second||past[i]<0||total[i]<=past[i]||total[i]>1024||size_t(past[i])!=lengths[size_t(rows[i])]||offsets[i]<0||offsets[i+1]<=offsets[i]||int64_t(offsets[i+1])-offsets[i]!=int64_t(total[i])-past[i]||int64_t(cu[i+1])-cu[i]!=total[i])throw std::invalid_argument("Kimi state row or logical length history differs");
  size_t first_page=1+(size_t(rows[i])-1)*4;
  for(size_t page=0;page<4;++page)if(table[i*4+page]!=int32_t(first_page+page))throw std::invalid_argument("Kimi Engine page table does not match owned physical state row");
  size_t count=size_t(total[i]-past[i]);for(size_t j=0;j<count;++j){size_t index=size_t(offsets[i])+j,position=size_t(past[i])+j;int64_t expected_slot=int64_t((first_page+position/256)*256+position%256);if(positions[index]!=int64_t(position)||slots[index]!=expected_slot)throw std::invalid_argument("Kimi logical position or physical cache slot differs");}
  requests.push_back({size_t(rows[i]),size_t(past[i]),count});
 }
 return requests;
}
inline void commit(std::vector<size_t>&lengths,const std::vector<Request>&requests){for(auto q:requests)lengths.at(q.row)=q.past+q.tokens;}
template<class Input>std::pair<size_t,size_t>max_lengths(const Input&a){
 auto offsets=values<int32_t>(a.input_offsets.value()),totals=values<int32_t>(a.total_sequence_lengths.value());size_t query=0,total=0;for(size_t i=0;i<totals.size();++i){query=std::max(query,size_t(offsets[i+1]-offsets[i]));total=std::max(total,size_t(totals[i]));}return {query,total};
}
}
