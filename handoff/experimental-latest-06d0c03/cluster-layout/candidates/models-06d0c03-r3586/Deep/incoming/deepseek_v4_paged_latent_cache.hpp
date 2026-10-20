#pragma once
#include "runtime/runtime.hpp"
#include <memory>
#include <vector>
namespace infinilm::models::deepseek_v4 {
// V4.0 physical pages: BF16 rotary tail, E4M3 NoPE and per64 E8M0 scales.
// The supplied Runtime owns allocation, stream, validation and lifetime.
class PagedLatentCache {
public:
 PagedLatentCache(runtime::Runtime&,size_t pages,size_t block=256);
 void store(runtime::Runtime&,const runtime::Tensor&slots,const runtime::Tensor&values);
 runtime::Tensor gather(runtime::Runtime&,const runtime::Tensor&slots)const;
 void reset(runtime::Runtime&);
 const runtime::Tensor&raw()const{return data_;}
 size_t stride()const{return stride_;}
private:
 std::vector<int64_t>validate(runtime::Runtime&,const runtime::Tensor&,bool)const;
 std::shared_ptr<runtime::Runtime>owner_;size_t pages_,block_,stride_;runtime::Tensor data_,flag_;std::vector<uint8_t>written_;
};
}
