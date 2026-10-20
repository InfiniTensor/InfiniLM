#include "glm_moe_dsa_indexer_selection.hpp"
#include "execution/submit.hpp"
#include <cmath>
#include <set>
#include <optional>
#include <stdexcept>
namespace infinilm::models::glm_moe_dsa {
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
IndexerSelection::IndexerSelection(rt::Runtime&r,size_t rows,size_t capacity,size_t heads,size_t dim,size_t topk,const engine::distributed::RankInfo&rank)
 :device_(r.device()),rows_(rows),capacity_(capacity),heads_(heads),local_heads_(0),dim_(dim),topk_(topk),lengths_(rows,0),tp_size_(rank.tp_size),comm_(rank.comm){
 r.assertOwnerThread();
 if(r.isGraphRecording()||device_.type()!=rt::Device::Type::kHygon||rank.device!=device_||!rows||!capacity||heads!=32||dim!=128||!topk||topk>2048||(tp_size_!=1&&tp_size_!=8)||(tp_size_>1&&!comm_))throw std::invalid_argument("GLM indexer selection requires explicit Hygon TP1/8 ownership and real geometry");
 local_heads_=heads/size_t(tp_size_);
 cache_=rt::Tensor::empty(r,{rows,capacity,dim},rt::DataType::kBFloat16);head_scale_=rt::Tensor::empty(r,{1},rt::DataType::kFloat32);
 float scale=1.f/std::sqrt(float(heads));head_scale_->copy_from(r,rt::Tensor::from_blob(&scale,{1},rt::DataType::kFloat32,{rt::Device::Type::kCpu,0}));r.syncStream();
}
void IndexerSelection::reset(rt::Runtime&r){r.assertOwnerThread();if(r.device()!=device_||r.isGraphRecording())throw std::invalid_argument("GLM indexer reset owner differs");r.syncStream();std::fill(lengths_.begin(),lengths_.end(),0);poisoned_=false;}
std::vector<IndexerSelection::Result> IndexerSelection::forward(rt::Runtime&r,const rt::Tensor&q,const rt::Tensor&k,const rt::Tensor&w,const std::vector<Request>&requests){
 r.assertOwnerThread();
 if(poisoned_||r.isGraphRecording()||r.device()!=device_||!q||!k||!w||q->ndim()!=3||q->size(1)!=local_heads_||q->size(2)!=dim_||!q->size(0)||k->shape()!=rt::Shape{q->size(0),dim_}||w->shape()!=rt::Shape{q->size(0),local_heads_}||q->dtype()!=rt::DataType::kBFloat16||k->dtype()!=rt::DataType::kBFloat16||w->dtype()!=rt::DataType::kFloat32||q->device()!=device_||k->device()!=device_||w->device()!=device_||requests.empty())throw std::invalid_argument("GLM indexer selection input/cache contract differs");
 size_t tokens=0;std::set<size_t> used;
 for(auto request:requests){if(request.row>=rows_||!used.insert(request.row).second||request.past!=lengths_[request.row]||!request.tokens||request.past>capacity_||request.tokens>capacity_-request.past||request.tokens>q->size(0)-tokens)throw std::invalid_argument("GLM indexer request row, lengths or capacity invalid");tokens+=request.tokens;}
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
 }r.syncStream();for(auto request:requests)lengths_[request.row]=request.past+request.tokens;return results;
 }catch(...){poisoned_=true;try{r.syncStream();}catch(...){std::terminate();}throw;}
}
}
