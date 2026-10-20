#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>
namespace infinilm::models::glm_moe_dsa {
struct SelectionMapR302 {std::vector<int64_t>logical,mapped;};
inline SelectionMapR302 selection_map_r302(const std::vector<int64_t>&host,size_t count,size_t width,size_t past) {
 if(!count||!width||count>std::numeric_limits<size_t>::max()/width||host.size()!=count*width||past>std::numeric_limits<size_t>::max()-count)
  throw std::invalid_argument("GLM selection mapping geometry differs");
 const size_t prefix=past+count;
 std::vector<uint8_t>used(prefix,0);
 for(size_t t=0;t<count;++t)for(size_t j=0;j<width;++j){const auto id=host[t*width+j];if(id<0||size_t(id)>=prefix)throw std::invalid_argument("GLM native topk outside valid prefix");if(size_t(id)<=past+t)used[size_t(id)]=1;}
 SelectionMapR302 result;std::vector<int64_t>mapping(prefix,-1);
 for(size_t id=0;id<prefix;++id)if(used[id]){mapping[id]=int64_t(result.logical.size());result.logical.push_back(int64_t(id));}
 result.mapped.resize(host.size());
 for(size_t t=0;t<count;++t){int64_t fallback=-1;for(size_t j=0;j<width;++j)if(size_t(host[t*width+j])<=past+t){fallback=mapping[size_t(host[t*width+j])];break;}if(fallback<0)throw std::invalid_argument("GLM native topk has no causal key");for(size_t j=0;j<width;++j){auto id=host[t*width+j];result.mapped[t*width+j]=size_t(id)<=past+t?mapping[size_t(id)]:fallback;}}
 return result;
}
}
