#include "deepseek_v4_paged_latent_cache.hpp"
#include "execution/submit.hpp"
#include <algorithm>
#include <set>
#include <stdexcept>
namespace infinilm::models::deepseek_v4 {
namespace rt=infinilm::runtime;
namespace ex=infinilm::execution;
namespace{
void invoke(rt::Runtime&r,const rt::Tensor&cache,const rt::Tensor&slots,const rt::Tensor&values,const rt::Tensor&status,size_t block,size_t mode){
 ex::submit<infini::ops::DeepseekV4PagedCache>(r,ex::RecordingPolicy::HostOnly,infini::ops::DeepseekV4PagedCache::DefaultConfig(r.device().type()),cache,slots,values,status,block,mode);
 int validation=0;r.memcpyD2H(&validation,status->data(),sizeof(validation));r.syncStream();
 if(validation)throw std::invalid_argument("DeepSeek native cache GPU validation status="+std::to_string(validation));
}
}
PagedLatentCache::PagedLatentCache(rt::Runtime&r,size_t pages,size_t block):owner_(r.shared_from_this()),pages_(pages),block_(block),stride_((block*584+575)/576*576){
 r.assertOwnerThread();if(r.isGraphRecording()||r.device().type()!=rt::Device::Type::kHygon||!pages||pages>65536||(block!=2&&block!=64&&block!=256))throw std::invalid_argument("DeepSeek native paged cache geometry differs");
 try{data_=rt::Tensor::empty(r,{pages_,stride_},rt::DataType::kUInt8);flag_=rt::Tensor::empty(r,{1},rt::DataType::kInt32);written_.assign(pages_*block_,0);r.setDeviceMemory(data_->data(),0xa5,data_->nbytes());r.syncStream();}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
std::vector<int64_t>PagedLatentCache::validate(rt::Runtime&r,const rt::Tensor&slots,bool write)const{
 r.assertOwnerThread();if(owner_.get()!=&r||r.isGraphRecording()||!slots||slots->device()!=r.device()||slots->dtype()!=rt::DataType::kInt64||slots->ndim()!=1||!slots->numel()||slots->numel()>65536||!slots->is_contiguous())throw std::invalid_argument("DeepSeek native cache slots/owner differ");
 std::vector<int64_t>host(slots->numel());r.memcpyD2H(host.data(),slots->data(),host.size()*8);std::set<int64_t>seen;
 for(auto slot:host){if(slot<0||size_t(slot)>=written_.size()||(write&&!seen.insert(slot).second)||(!write&&!written_[size_t(slot)]))throw std::invalid_argument("DeepSeek native cache invalid/duplicate write or unwritten read slot");}return host;
}
void PagedLatentCache::store(rt::Runtime&r,const rt::Tensor&slots,const rt::Tensor&values){
 auto host=validate(r,slots,true);if(!values||values->device()!=r.device()||values->dtype()!=rt::DataType::kBFloat16||values->shape()!=rt::Shape{host.size(),512}||!values->is_contiguous())throw std::invalid_argument("DeepSeek native cache requires contiguous BF16 D512 values");
 try{invoke(r,data_,slots,values,flag_,block_,0);for(auto slot:host)written_[size_t(slot)]=1;}
 catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
rt::Tensor PagedLatentCache::gather(rt::Runtime&r,const rt::Tensor&slots)const{
 auto host=validate(r,slots,false);auto out=rt::Tensor::empty(r,{host.size(),512},rt::DataType::kBFloat16);try{invoke(r,data_,slots,out,flag_,block_,1);return out;}catch(...){try{r.syncStream();}catch(...){std::terminate();}throw;}
}
void PagedLatentCache::reset(rt::Runtime&r){r.assertOwnerThread();if(owner_.get()!=&r||r.isGraphRecording())throw std::invalid_argument("DeepSeek native cache reset owner differs");r.syncStream();std::fill(written_.begin(),written_.end(),0);}
}
