#pragma once
#include <cstdint>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>
namespace infinilm::models::native_sparse {
struct Request{size_t row,past,tokens;};
inline std::vector<Request>validate_requests(size_t rows,size_t capacity,const std::vector<size_t>&lengths,const std::vector<int32_t>&physical,const std::vector<int32_t>&past,const std::vector<int32_t>&total,const std::vector<int32_t>&offsets,size_t tokens){
 size_t n=physical.size();if(!n||n>rows||lengths.size()!=rows||past.size()!=n||total.size()!=n||offsets.size()!=n+1||offsets[0]!=0||tokens>4096||offsets.back()<0||size_t(offsets.back())!=tokens)throw std::invalid_argument("Native sparse packed request geometry differs");
 std::set<size_t>seen;std::vector<Request>result;result.reserve(n);
 for(size_t i=0;i<n;++i){if(physical[i]<0||size_t(physical[i])>=rows||!seen.insert(size_t(physical[i])).second||past[i]<0||total[i]<past[i]||offsets[i]<0||offsets[i+1]<=offsets[i]||size_t(past[i])!=lengths[size_t(physical[i])]||size_t(total[i])>capacity||int64_t(total[i])-past[i]!=int64_t(offsets[i+1])-offsets[i])throw std::invalid_argument("Native sparse row/past/total/offset metadata differs");result.push_back({size_t(physical[i]),size_t(past[i]),size_t(total[i]-past[i])});}
 return result;
}
inline void commit_lengths(std::vector<size_t>&lengths,const std::vector<Request>&requests){for(auto q:requests)lengths[q.row]=q.past+q.tokens;}
}
