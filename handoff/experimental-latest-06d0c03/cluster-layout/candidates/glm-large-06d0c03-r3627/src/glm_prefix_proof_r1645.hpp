#pragma once
#include "runtime/runtime.hpp"
#include <memory>
#include <stdexcept>
#include <vector>
namespace infinilm::models::glm_moe_dsa {
class IndexerSelectionR1581;
// Private capability for this selector's immutable ascending [0,prefix) view.
// Only the producing selector constructs it. It is not an authorization boundary
// for arbitrary device writes; all consumers must honor the read-only contract.
class PrefixProofR1645 final {
 std::shared_ptr<runtime::Runtime>owner_;runtime::Tensor value_;
 size_t row_,past_,capacity_;
 friend class IndexerSelectionR1581;
 PrefixProofR1645(runtime::Runtime&r,const runtime::Tensor&value,size_t row,size_t past,size_t capacity):owner_(r.shared_from_this()),value_(value),row_(row),past_(past),capacity_(capacity){
  r.assertOwnerThread();
  if(r.isGraphRecording()||!value||value->device()!=r.device()||value->dtype()!=runtime::DataType::kInt64||value->shape()!=runtime::Shape{1,past+1}||!value->is_contiguous()||past>=capacity||capacity>2048)throw std::invalid_argument("1645 invalid immutable prefix producer");
 }
public:
 PrefixProofR1645(const PrefixProofR1645&)=delete;PrefixProofR1645&operator=(const PrefixProofR1645&)=delete;
 size_t capacity()const noexcept{return capacity_;}
 bool matches(runtime::Runtime&r,const runtime::Tensor&value,size_t row,size_t past,size_t tokens,size_t capacity)const{
  r.assertOwnerThread();if(owner_.get()!=&r||r.isGraphRecording())throw std::invalid_argument("1645 foreign Runtime or graph consumer");
  return tokens==1&&row==row_&&past==past_&&capacity==capacity_&&past<capacity&&value&&value.operator->()==value_.operator->()&&value->data()==value_->data()&&value->device()==r.device()&&value->dtype()==runtime::DataType::kInt64&&value->shape()==runtime::Shape{1,past+1}&&value->is_contiguous();
 }
};
using PrefixProofsR1645=std::vector<std::shared_ptr<const PrefixProofR1645>>;
struct ProvenPrefixR1645 {std::vector<runtime::Tensor>values;PrefixProofsR1645 proofs;};
}
