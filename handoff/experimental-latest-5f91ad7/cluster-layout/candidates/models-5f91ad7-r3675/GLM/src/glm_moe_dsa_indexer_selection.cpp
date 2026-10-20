#include <array>
#include <atomic>
#include <limits>
#include <numeric>
#include "glm_moe_dsa_indexer_selection.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <set>
#include <optional>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{std::array<std::atomic<uint64_t>,8>all_calls980{},full_calls980{},metadata_calls1406{};void count980(int d,bool fast){if(d>=0&&d<8)(fast?all_calls980:full_calls980)[size_t(d)].fetch_add(1,std::memory_order_relaxed);}}
IndexerSelectionR1581::IndexerSelectionR1581(rt::Runtime&r,size_t rows,size_t capacity,size_t heads,size_t dim,size_t topk,const engine::distributed::RankInfo&rank)
 :owner_(&r),missing_keys_r1406_(rows,false),device_(r.device()),rows_(rows),capacity_(capacity),heads_(heads),local_heads_(0),dim_(dim),topk_(topk),lengths_(rows,0),tp_size_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||!rows||!capacity||heads!=32||dim!=128||!topk||topk>2048||(tp_size_!=1&&tp_size_!=8)||(tp_size_>1&&!comm_))throw std::invalid_argument("GLM indexer selection requires explicit Hygon TP1/8 ownership and real geometry");
 local_heads_=heads/size_t(tp_size_);
 cache_=rt::Tensor::empty(r,{rows,capacity,dim},rt::DataType::kBFloat16);head_scale_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);
 float scale=1.f/std::sqrt(float(heads));head_scale_->copy_from(r,rt::Tensor::from_blob(&scale,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}));r.syncStream();
}
void IndexerSelectionR1581::reset(rt::Runtime&r){r.assertOwnerThread();if(&r!=owner_||r.device()!=device_||r.isGraphRecording())throw std::invalid_argument("GLM indexer reset owner differs");r.syncStream();std::fill(lengths_.begin(),lengths_.end(),0);std::fill(missing_keys_r1406_.begin(),missing_keys_r1406_.end(),false);poisoned_=false;}
std::vector<IndexerSelectionR1581::Result> IndexerSelectionR1581::forward(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&w,const std::vector<Request>&requests){
 r.assertOwnerThread();
 if(poisoned_||r.isGraphRecording()||&r!=owner_||r.device()!=device_||!q||!k||!w||q->ndim()!=3||q->size(1)!=local_heads_||q->size(2)!=dim_||!q->size(0)||k->shape()!=rt::Shape{q->size(0),dim_}||w->shape()!=rt::Shape{q->size(0),local_heads_}||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=rt::DataType::kBFloat16||w->dtype()!=rt::DataType::kFloat32||q->device()!=device_||k->device()!=device_||w->device()!=device_||requests.empty())throw std::invalid_argument("GLM indexer selection input/cache contract differs");
 size_t tokens=0;std::set<size_t> used;
 for(auto request:requests){if(request.row>=rows_||!used.insert(request.row).second||request.past!=lengths_[request.row]||!request.tokens||request.past>capacity_||request.tokens>capacity_-request.past||request.tokens>q->size(0)-tokens)throw std::invalid_argument("GLM indexer request row, lengths or capacity invalid");if(missing_keys_r1406_[request.row])throw std::logic_error("1406 incomplete indexer cache requires reset before scoring");tokens+=request.tokens;}
 if(tokens!=q->size(0))throw std::invalid_argument("GLM indexer packed request lengths differ");
 auto alloc=[&](rt::Shape shape,rt::DataType dt=rt::DataType::kFloat32){return rt::Tensor::empty(r,shape,dt);};
 std::vector<Result> results;size_t offset=0;
 try{for(auto request:requests){size_t t=request.tokens,total=request.past+t;
  cache_->narrow({{0,request.row,1},{1,request.past,t}})->view({t,dim_})->copy_from(r,k->narrow({{0,offset,t}}));
  auto keys=cache_->narrow({{0,request.row,1},{1,0,total}})->view({total,dim_});
  auto qf=alloc({t*local_heads_,dim_}),kf=alloc({total,dim_});
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),q->narrow({{0,offset,t}})->view({t*local_heads_,dim_}),qf);
  ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(device_.type()),keys,kf);
  auto dots=alloc({t*local_heads_,total}),relu=alloc({t*local_heads_,total});
  ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(device_.type()),qf,kf->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1.f/std::sqrt(float(dim_))},std::optional<float>{0},std::optional<int>{},std::optional<int>{},dots);
  ex::submit<infini::ops::Relu>(r,infini::ops::Relu::DefaultConfig(device_.type()),dots,relu);
  auto scaled_weights=alloc({t,local_heads_}),weighted=alloc({t,local_heads_,total}),scores=alloc({t,total});
  ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),w->narrow({{0,offset,t}}),head_scale_,scaled_weights);
  ex::submit<infini::ops::Mul>(r,infini::ops::Mul::DefaultConfig(device_.type()),relu->view({t,local_heads_,total}),scaled_weights->view({t,local_heads_,1}),weighted);
  ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(device_.type()),weighted,std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},scores);
  if(tp_size_>1)engine::distributed::allreduce_(r,scores,scores,infinicclSum,comm_);
  std::vector<float> host_mask(t*total);for(size_t i=0;i<t;++i)for(size_t j=request.past+i+1;j<total;++j)host_mask[i*total+j]=-INFINITY;
  auto mask=alloc({t,total}),masked=alloc({t,total});mask->copy_from(r,rt::Tensor::from_blob(host_mask.data(),{t,total},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}));r.syncStream();
  ex::submit<infini::ops::Add>(r,infini::ops::Add::DefaultConfig(device_.type()),scores,mask,masked);
  size_t selected=std::min(topk_,total);auto values=alloc({t,selected}),indices=alloc({t,selected},rt::DataType::kInt64);
  ex::submit<infini::ops::Topk>(r,infini::ops::Topk::DefaultConfig(device_.type()),masked,int64_t(selected),int64_t{1},true,true,values,indices);
  results.push_back({masked,indices});offset+=t;
 }r.syncStream();for(auto request:requests)lengths_[request.row]=request.past+request.tokens;count980(r.device().index(),false);return results;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
std::vector<rt::Tensor> IndexerSelectionR1581::select_all(rt::Runtime&r,const rt::Tensor&key,const std::vector<Request>&requests){
 r.assertOwnerThread();
 if(poisoned_||r.isGraphRecording()||&r!=owner_||r.device()!=device_||!key||key->device()!=device_||key->dtype()!=rt::DataType::kBFloat16||key->ndim()!=2||key->size(1)!=dim_||!key->size(0)||!key->is_contiguous()||requests.empty()||requests.size()!=key->size(0))throw std::invalid_argument("980all-selected decode key/owner contract");
 std::set<size_t>seen;size_t entries=0;
 for(auto request:requests){
  if(request.row>=rows_||!seen.insert(request.row).second||request.tokens!=1||request.past!=lengths_[request.row]||request.past>=capacity_||request.past+1>topk_||request.past+1>std::numeric_limits<size_t>::max()-entries)throw std::invalid_argument("980all-selected decode row/length/topk contract");
  if(missing_keys_r1406_[request.row])throw std::logic_error("1406 incomplete indexer cache requires reset before key-updating select_all");
  entries+=request.past+1;
 }
 // These owners survive the catch-path stream drain, including asynchronous H2D.
 std::vector<int64_t>host;std::vector<rt::Tensor>results;rt::Tensor bank;
 try{
  host.resize(entries);size_t offset=0;for(auto request:requests){std::iota(host.begin()+offset,host.begin()+offset+request.past+1,int64_t{0});offset+=request.past+1;}
  bank=rt::Tensor::empty(r,{entries},rt::DataType::kInt64);bank->copy_from(r,rt::Tensor::from_blob(host.data(),{entries},rt::DataType::kInt64,{rt::Device::Type::kCpu,0}));offset=0;
  for(size_t i=0;i<requests.size();++i){auto request=requests[i];cache_->narrow({{0,request.row,1},{1,request.past,1}})->view({1,dim_})->copy_from(r,key->narrow({{0,i,1}}));size_t prefix=request.past+1;results.push_back(bank->narrow({{0,offset,prefix}})->view({1,prefix}));offset+=prefix;}
  r.syncStream();for(auto request:requests)lengths_[request.row]=request.past+1;count980(r.device().index(),true);return results;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
std::vector<rt::Tensor> IndexerSelectionR1581::select_all_metadata(rt::Runtime&r,const std::vector<Request>&requests){
 r.assertOwnerThread();
 if(&r!=owner_||poisoned_||r.isGraphRecording()||r.device()!=device_||capacity_>topk_||requests.empty())throw std::invalid_argument("1406 metadata-only bounded owner/capacity/state contract");
 std::set<size_t>seen;size_t entries=0;
 for(auto request:requests){if(request.row>=rows_||!seen.insert(request.row).second||request.tokens!=1||request.past!=lengths_[request.row]||request.past>=capacity_||request.past+1>topk_||request.past+1>std::numeric_limits<size_t>::max()-entries)throw std::invalid_argument("1406 metadata-only row/past/tokens/topk contract");entries+=request.past+1;}
 // No scores or indexer keys are fabricated. This mode is valid only while the
 // configured context is bounded by top-k; stale scoring is rejected until reset.
 std::vector<int64_t>host;std::vector<rt::Tensor>results;rt::Tensor bank;
 try{host.resize(entries);size_t offset=0;for(auto request:requests){std::iota(host.begin()+offset,host.begin()+offset+request.past+1,int64_t{0});offset+=request.past+1;}
  bank=rt::Tensor::empty(r,{entries},rt::DataType::kInt64);bank->copy_from(r,rt::Tensor::from_blob(host.data(),{entries},rt::DataType::kInt64,{rt::Device::Type::kCpu,0}));offset=0;
  for(auto request:requests){size_t prefix=request.past+1;results.push_back(bank->narrow({{0,offset,prefix}})->view({1,prefix}));offset+=prefix;}
  r.syncStream();for(auto request:requests){lengths_[request.row]=request.past+1;missing_keys_r1406_[request.row]=true;}count980(r.device().index(),true);metadata_calls1406.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);return results;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}

std::vector<rt::Tensor> IndexerSelectionR1581::select_all_metadata_borrowed_readonly(rt::Runtime&r,const std::vector<Request>&requests){
 r.assertOwnerThread();
 if(&r!=owner_||poisoned_||r.isGraphRecording()||r.device()!=device_||capacity_>topk_||requests.empty())throw std::invalid_argument("1581 borrowed readonly bounded owner/capacity/state contract");
 std::set<size_t>seen;
 for(auto request:requests)if(request.row>=rows_||!seen.insert(request.row).second||request.tokens!=1||request.past!=lengths_[request.row]||request.past>=capacity_||request.past+1>topk_)throw std::invalid_argument("1581 borrowed readonly row/past/tokens/topk contract");
 std::vector<rt::Tensor>results;std::vector<int64_t>host;rt::Tensor building;
 try{
  if(!prefix_bank_r1581_){
   if(rows_>std::numeric_limits<size_t>::max()/capacity_||rows_*capacity_>std::numeric_limits<size_t>::max()/sizeof(int64_t))throw std::overflow_error("1581 bank size");
   host.resize(rows_*capacity_);for(size_t row=0;row<rows_;++row)std::iota(host.begin()+row*capacity_,host.begin()+(row+1)*capacity_,int64_t{0});
   building=rt::Tensor::empty(r,{rows_,capacity_},rt::DataType::kInt64);building->copy_from(r,rt::Tensor::from_blob(host.data(),{rows_,capacity_},rt::DataType::kInt64,{rt::Device::Type::kCpu,0}));r.syncStream();prefix_bank_r1581_=std::move(building);++prefix_stats_r1581_.bank_builds;
  }
  results.reserve(requests.size());for(auto request:requests)results.push_back(prefix_bank_r1581_->narrow({{0,request.row,1},{1,0,request.past+1}})->squeeze(0)->view({1,request.past+1}));
  for(auto request:requests){lengths_[request.row]=request.past+1;missing_keys_r1406_[request.row]=true;}
  ++prefix_stats_r1581_.calls;prefix_stats_r1581_.returned_views+=results.size();count980(r.device().index(),true);metadata_calls1406.at(r.device().index()).fetch_add(1,std::memory_order_relaxed);return results;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}

ProvenPrefixR1645 IndexerSelectionR1581::select_all_metadata_proven_readonly(rt::Runtime&r,const std::vector<Request>&requests){
 r.assertOwnerThread();if(&r!=owner_||r.isGraphRecording())throw std::invalid_argument("1646 wrong producer owner/graph");
 ProvenPrefixR1645 result;result.values=select_all_metadata_borrowed_readonly(r,requests);
 try{result.proofs.reserve(requests.size());
  for(size_t i=0;i<requests.size();++i){auto q=requests[i];result.proofs.emplace_back(new PrefixProofR1645(r,result.values.at(i),q.row,q.past,capacity_));}return result;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}

}
extern "C" uint64_t modern_glm_indexer_select_all_calls_r980(int device,int fast){if(device<0||device>=8||fast<0||fast>1)return 0;return(fast?infinilm::models::glm_moe_dsa::all_calls980:infinilm::models::glm_moe_dsa::full_calls980)[size_t(device)].load(std::memory_order_relaxed);}

extern "C" uint64_t modern_glm_indexer_metadata_calls_r1406(int device){if(device<0||device>=8)return 0;return infinilm::models::glm_moe_dsa::metadata_calls1406[size_t(device)].load(std::memory_order_relaxed);}
