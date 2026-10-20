#pragma once
#include "native_sparse_cache_r22.hpp"
#include "native_sparse_requests_r22.hpp"
#include "runtime/tensor.hpp"
#include <cstring>
namespace infinilm::models::native_sparse {
namespace rt=infinilm::runtime;
inline rt::Tensor freeze_cpu(const std::optional<rt::Tensor>&value,rt::DataType dtype,bool one_dim){
 if(!value||!value.value()||value.value()->device().type()!=rt::Device::Type::kCpu||value.value()->dtype()!=dtype||!value.value()->is_contiguous()||(one_dim&&value.value()->ndim()!=1))throw std::invalid_argument("Native sparse Engine requires contiguous original CPU input metadata");
 auto&t=value.value();auto owner=std::make_shared<std::vector<std::byte>>(t->nbytes());if(t->nbytes())std::memcpy(owner->data(),t->data(),t->nbytes());return rt::Tensor::from_blob(owner->data(),t->shape(),dtype,{rt::Device::Type::kCpu,0},owner);
}
template<class T>std::vector<T>values(const rt::Tensor&t){const auto*p=reinterpret_cast<const T*>(t->data());return {p,p+t->numel()};}
template<class Input>std::vector<Request>freeze_input(Input&input,const cache::NativeSparseCacheConfig&cache,const std::vector<size_t>&lengths,size_t vocab){
 if(input.pixel_values||input.target_hidden_states||input.image_grid_thw||input.block_tables||input.slot_mapping)throw std::invalid_argument("Native sparse Engine owns its typed caches and accepts text only");
 input.input_ids=freeze_cpu(input.input_ids,rt::DataType::kInt64,false);input.position_ids=freeze_cpu(input.position_ids,rt::DataType::kInt64,false);
 if(input.input_ids.value()->ndim()!=2||input.input_ids.value()->size(0)!=1||input.position_ids.value()->numel()!=input.input_ids.value()->numel())throw std::invalid_argument("Native sparse requires packed [1,T] input IDs and matching positions");
 input.past_sequence_lengths=freeze_cpu(input.past_sequence_lengths,rt::DataType::kInt32,true);input.total_sequence_lengths=freeze_cpu(input.total_sequence_lengths,rt::DataType::kInt32,true);input.input_offsets=freeze_cpu(input.input_offsets,rt::DataType::kInt32,true);
 input.mamba_init_state_indices=freeze_cpu(input.mamba_init_state_indices,rt::DataType::kInt32,true);input.mamba_final_state_indices=freeze_cpu(input.mamba_final_state_indices,rt::DataType::kInt32,true);
 auto rows=values<int32_t>(input.mamba_init_state_indices.value());if(rows!=values<int32_t>(input.mamba_final_state_indices.value()))throw std::invalid_argument("Native sparse state read/write rows differ");
 auto requests=validate_requests(cache.rows(),cache.capacity(),lengths,rows,values<int32_t>(input.past_sequence_lengths.value()),values<int32_t>(input.total_sequence_lengths.value()),values<int32_t>(input.input_offsets.value()),input.input_ids.value()->numel());
 auto ids=values<int64_t>(input.input_ids.value()),positions=values<int64_t>(input.position_ids.value());
 for(auto id:ids)if(id<0||size_t(id)>=vocab)throw std::invalid_argument("Native sparse input token outside vocabulary");
 size_t offset=0;for(auto q:requests){for(size_t j=0;j<q.tokens;++j)if(positions[offset+j]!=int64_t(q.past+j))throw std::invalid_argument("Native sparse input position differs from logical history");offset+=q.tokens;}
 return requests;
}
}
