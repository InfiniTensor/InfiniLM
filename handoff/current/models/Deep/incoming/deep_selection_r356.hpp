#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
namespace infinilm::models::deepseek_v4 {
struct LatentSelectionR356 {
 std::vector<std::vector<int64_t>> selected;
 std::vector<int64_t> packed_ids;
 std::vector<size_t> packed_offsets;
};
inline LatentSelectionR356 latent_selection_r356(const std::vector<int64_t>&hs,
 const std::vector<int64_t>&hc,const std::vector<int32_t>&hl,size_t tokens,
 size_t sw,size_t cw,size_t swa_rows,size_t compressed_rows){
 const auto limit=std::numeric_limits<size_t>::max();
 if(!sw||!cw||tokens>limit/2||tokens>limit/sw||tokens>limit/cw||
    hs.size()!=tokens*sw||hc.size()!=tokens*cw||hl.size()!=tokens*2||
    swa_rows>size_t(INT64_MAX)||compressed_rows>size_t(INT64_MAX)-swa_rows)
  throw std::invalid_argument("DeepSeek latent host selection shape/size overflows");
 LatentSelectionR356 result;result.selected.resize(tokens);result.packed_offsets.resize(tokens+1,0);
 const size_t span=swa_rows+compressed_rows;
 // Actual bounded model banks are small. Preserve sparse behavior for huge
 // caller-owned banks rather than allocating proportional to an unbounded ID.
 std::vector<size_t>generation;if(span<=1048576)generation.assign(span,0);
 for(size_t t=0;t<tokens;++t){
  if(hl[2*t]<0||hl[2*t+1]<0||size_t(hl[2*t])>sw||size_t(hl[2*t+1])>cw)
   throw std::invalid_argument("DeepSeek selected lengths invalid");
  std::set<int64_t>sparse;
  for(int bank=0;bank<2;++bank){
   size_t width=bank?cw:sw,rows=bank?compressed_rows:swa_rows;
   const auto&host=bank?hc:hs;
   for(int j=0;j<hl[2*t+bank];++j){
    int64_t id=host[t*width+size_t(j)];
    if(id<0||size_t(id)>=rows)throw std::invalid_argument("DeepSeek selected KV index invalid");
    id+=bank?int64_t(swa_rows):0;
    bool duplicate;
    if(!generation.empty()){duplicate=generation[size_t(id)]==t+1;generation[size_t(id)]=t+1;}
    else duplicate=!sparse.insert(id).second;
    if(duplicate)throw std::invalid_argument("DeepSeek duplicate selected KV index");
    result.selected[t].push_back(id);
   }
  }
 }
 // Preserve the original bank order, per-query selection and host-copy order.
 for(size_t t=0;t<tokens;++t){result.packed_ids.insert(result.packed_ids.end(),result.selected[t].begin(),result.selected[t].end());result.packed_offsets[t+1]=result.packed_ids.size();}
 return result;
}
}
