#include "execution/submit.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <optional>
extern "C" int modern_grouped_decode_batch(const void*,const void*,const void*,void*,size_t,size_t,size_t,void*);
extern "C" int modern_grouped_scatter_batch(void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int modern_grouped_decode(const void*,const void*,void*,size_t,size_t,void*);
extern "C" int modern_grouped_activate(const void*,void*,size_t,size_t,void*);
extern "C" int modern_grouped_scatter(void*,const void*,const void*,const void*,size_t,size_t,void*);
extern "C" int modern_grouped_decode_batch_f32_r21(const void*,const void*,const void*,void*,size_t,size_t,size_t,void*);
extern "C" int modern_grouped_gather_f32_r21(const void*,const void*,void*,size_t,size_t,void*);
extern "C" int modern_grouped_activate_f32_r21(const void*,void*,size_t,size_t,void*);
namespace rt=infinilm::runtime;namespace ex=infinilm::execution;
namespace{thread_local std::string error;thread_local std::shared_ptr<rt::Runtime> supplied_runtime;thread_local size_t plan_info[5]={};
std::map<int,std::shared_ptr<rt::Runtime>>&worker_owners(){static thread_local std::map<int,std::shared_ptr<rt::Runtime>>pool;return pool;}
std::shared_ptr<rt::Runtime>worker(int device){if(supplied_runtime){supplied_runtime->assertOwnerThread();if(supplied_runtime->device()!=rt::Device{rt::Device::Type::kHygon,device})throw std::invalid_argument("Grouped supplied Runtime owner differs");return supplied_runtime;}auto&r=worker_owners()[device];if(!r){r=rt::Runtime::create({rt::Device::Type::kHygon,device});r->enableOwnerDrainFailStop();}return r;}
struct Work{std::shared_ptr<rt::Runtime>r;std::vector<rt::Tensor>held;std::vector<std::shared_ptr<void>>host_owners;~Work(){if(r)try{r->syncStream();}catch(...){std::terminate();}}rt::Tensor alloc(rt::Shape shape,rt::DataType dt){auto t=rt::Tensor::empty(*r,shape,dt);held.push_back(t);return t;}};
void check(int code){if(code)throw std::runtime_error(cudaGetErrorString(static_cast<cudaError_t>(code)));}
}
extern "C" const char*modern_grouped_error(){return error.c_str();}
extern "C" int modern_grouped_mxfp4(int mode,int device,size_t tokens,size_t experts,size_t hidden,size_t intermediate,size_t topk,
 void*output,const void*input,const void*indices,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2,void*stream){
 Work work;try{
 if((mode!=0&&mode!=1&&mode!=2&&mode!=3)||!tokens||tokens>6144||!experts||experts>896||!hidden||hidden%32||!intermediate||intermediate%32||!topk||topk>16||topk>experts)throw std::invalid_argument("Grouped MXFP4 qualification geometry invalid");
 infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};check(int(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))));work.r=worker(device);auto&r=*work.r;
 auto view=[&](const void*p,rt::Shape shape,rt::DataType dt){auto t=rt::Tensor::from_blob(const_cast<void*>(p),shape,dt,r.device());work.held.push_back(t);return t;};
 auto x=view(input,{tokens,hidden},rt::DataType::kBFloat16),out=view(output,{tokens,hidden},rt::DataType::kBFloat16),ids=view(indices,{tokens,topk},rt::DataType::kInt32),rw=view(weights,{tokens,topk},rt::DataType::kFloat32);
 auto p13=view(w13,{experts,2*intermediate,hidden/2},rt::DataType::kUInt8),sc13=view(s13,{experts,2*intermediate,hidden/32},rt::DataType::kUInt8),p2=view(w2,{experts,hidden,intermediate/2},rt::DataType::kUInt8),sc2=view(s2,{experts,hidden,intermediate/32},rt::DataType::kUInt8);
 if(mode==0){auto config=infini::ops::FusedMoeMxfp4::DefaultConfig(r.device().type());config.set_implementation_index(1);ex::submit<infini::ops::FusedMoeMxfp4>(r,ex::RecordingPolicy::HostOnly,config,out,x,ids,rw,p13,sc13,p2,sc2,size_t{2});r.syncStream();return 0;}
 // Route metadata alone visits the host; all numerical math remains on the DCU.
 std::vector<int32_t>host_ids(tokens*topk);r.syncStream();r.memcpyD2H(host_ids.data(),ids->data(),host_ids.size()*4);
 struct Group{std::vector<int64_t>rows,routes;};auto group_owner=std::make_shared<std::vector<Group>>(experts);work.host_owners.push_back(group_owner);auto&groups=*group_owner;size_t max_rows=0;
 for(size_t route=0;route<host_ids.size();++route){int expert=host_ids[route];if(expert<0||size_t(expert)>=experts)throw std::invalid_argument("Grouped MXFP4 expert ID outside bank");auto&g=groups[size_t(expert)];g.rows.push_back(int64_t(route/topk));g.routes.push_back(int64_t(route));max_rows=std::max(max_rows,g.rows.size());}
 if(mode==2||mode==3){
  struct Plan{size_t padded;std::vector<int32_t>experts;std::vector<int64_t>rows,routes;};
  std::map<size_t,std::vector<size_t>>bins;for(size_t expert=0;expert<experts;++expert)if(!groups[expert].rows.empty())bins[(groups[expert].rows.size()+7)/8*8].push_back(expert);
  auto plans_owner=std::make_shared<std::vector<Plan>>();work.host_owners.push_back(plans_owner);auto&plans=*plans_owner;size_t max_count=0,max_padded=0;
  for(auto&bin:bins){size_t padded=bin.first;for(size_t begin=0;begin<bin.second.size();begin+=32){size_t count=std::min<size_t>(32,bin.second.size()-begin);Plan plan;plan.padded=padded;plan.rows.assign(count*padded,0);plan.routes.assign(count*padded,-1);
    for(size_t j=0;j<count;++j){size_t expert=bin.second[begin+j];plan.experts.push_back(int32_t(expert));std::copy(groups[expert].rows.begin(),groups[expert].rows.end(),plan.rows.begin()+j*padded);std::copy(groups[expert].routes.begin(),groups[expert].routes.end(),plan.routes.begin()+j*padded);}
    plans.push_back(std::move(plan));max_count=std::max(max_count,count);max_padded=std::max(max_padded,padded);
  }}
  auto d13f=work.alloc({max_count,2*intermediate,hidden},rt::DataType::kFloat32),d2f=work.alloc({max_count,hidden,intermediate},rt::DataType::kFloat32);
  size_t actual_capacity=0;for(const auto&plan:plans)actual_capacity=std::max(actual_capacity,plan.experts.size()*plan.padded);
  size_t capacity=mode==3?max_count*max_padded:actual_capacity;plan_info[0]=max_count;plan_info[1]=max_padded;plan_info[2]=max_count*max_padded;plan_info[3]=actual_capacity;plan_info[4]=capacity;auto row_ids=work.alloc({capacity},rt::DataType::kInt64),route_ids=work.alloc({capacity},rt::DataType::kInt64),expert_ids=work.alloc({max_count},rt::DataType::kInt32);
  auto gu=work.alloc({capacity,2*intermediate},rt::DataType::kFloat32),go=work.alloc({capacity,hidden},rt::DataType::kFloat32),contributions=work.alloc({tokens*topk,hidden},rt::DataType::kFloat32);
  auto gxf=work.alloc({capacity,hidden},rt::DataType::kFloat32),actf=work.alloc({capacity,intermediate},rt::DataType::kFloat32);
  auto gemm=[&](const rt::Tensor&a,const rt::Tensor&b,const rt::Tensor&c){ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),a,b->permute({0,2,1}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},c);};
  for(auto&plan:plans){size_t count=plan.experts.size(),padded=plan.padded,rows=count*padded;
   r.memcpyH2D(row_ids->data(),plan.rows.data(),rows*sizeof(int64_t),true);r.memcpyH2D(route_ids->data(),plan.routes.data(),rows*sizeof(int64_t),true);r.memcpyH2D(expert_ids->data(),plan.experts.data(),count*sizeof(int32_t),true);
   auto group_x=gxf->narrow({{0,0,rows}}),gate_up=gu->narrow({{0,0,rows}}),activated=actf->narrow({{0,0,rows}}),group_out=go->narrow({{0,0,rows}});
   check(modern_grouped_gather_f32_r21(input,row_ids->data(),group_x->data(),rows,hidden,r.stream()));
   check(modern_grouped_decode_batch_f32_r21(w13,s13,expert_ids->data(),d13f->data(),count,2*intermediate,hidden,r.stream()));
   gemm(group_x->view({count,padded,hidden}),d13f->narrow({{0,0,count}}),gate_up->view({count,padded,2*intermediate}));
   check(modern_grouped_activate_f32_r21(gate_up->data(),activated->data(),rows,intermediate,r.stream()));
   check(modern_grouped_decode_batch_f32_r21(w2,s2,expert_ids->data(),d2f->data(),count,hidden,intermediate,r.stream()));
   gemm(activated->view({count,padded,intermediate}),d2f->narrow({{0,0,count}}),group_out->view({count,padded,hidden}));
   check(modern_grouped_scatter_batch(contributions->data(),group_out->data(),route_ids->data(),weights,rows,hidden,r.stream()));
  }
  auto sum=work.alloc({tokens,hidden},rt::DataType::kFloat32);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),contributions->view({tokens,topk,hidden}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},sum);ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),sum,out);r.syncStream();return 0;
 }
 auto d13=work.alloc({2*intermediate,hidden},rt::DataType::kBFloat16),d2=work.alloc({hidden,intermediate},rt::DataType::kBFloat16);
 auto row_ids=work.alloc({max_rows},rt::DataType::kInt64),route_ids=work.alloc({max_rows},rt::DataType::kInt64),gx=work.alloc({max_rows,hidden},rt::DataType::kBFloat16),gu=work.alloc({max_rows,2*intermediate},rt::DataType::kFloat32),act=work.alloc({max_rows,intermediate},rt::DataType::kBFloat16),go=work.alloc({max_rows,hidden},rt::DataType::kFloat32),contributions=work.alloc({tokens*topk,hidden},rt::DataType::kFloat32);
 auto gemm=[&](const rt::Tensor&a,const rt::Tensor&b,const rt::Tensor&c){ex::submit<infini::ops::Gemm>(r,infini::ops::Gemm::DefaultConfig(r.device().type()),a,b->permute({1,0}),std::optional<rt::Tensor>{},std::optional<float>{1},std::optional<float>{0},std::optional<int>{},std::optional<int>{},c);};
 for(size_t expert=0;expert<experts;++expert){auto&g=groups[expert];size_t count=g.rows.size();if(!count)continue;
  auto upload=[&](const std::vector<int64_t>&v,const rt::Tensor&dst){auto result=dst->narrow({{0,0,count}});r.memcpyH2D(result->data(),v.data(),count*sizeof(int64_t),true);return result;};
  auto rows=upload(g.rows,row_ids),routes=upload(g.routes,route_ids),group_x=gx->narrow({{0,0,count}}),gate_up=gu->narrow({{0,0,count}}),activated=act->narrow({{0,0,count}}),group_out=go->narrow({{0,0,count}});
  ex::submit<infini::ops::IndexSelect>(r,infini::ops::IndexSelect::DefaultConfig(r.device().type()),x,rows,int64_t{0},group_x);
  check(modern_grouped_decode(static_cast<const unsigned char*>(w13)+expert*2*intermediate*(hidden/2),static_cast<const unsigned char*>(s13)+expert*2*intermediate*(hidden/32),d13->data(),2*intermediate,hidden,r.stream()));
  gemm(group_x,d13,gate_up);check(modern_grouped_activate(gate_up->data(),activated->data(),count,intermediate,r.stream()));
  check(modern_grouped_decode(static_cast<const unsigned char*>(w2)+expert*hidden*(intermediate/2),static_cast<const unsigned char*>(s2)+expert*hidden*(intermediate/32),d2->data(),hidden,intermediate,r.stream()));
  gemm(activated,d2,group_out);check(modern_grouped_scatter(contributions->data(),group_out->data(),routes->data(),weights,count,hidden,r.stream()));
 }
 auto sum=work.alloc({tokens,hidden},rt::DataType::kFloat32);ex::submit<infini::ops::Sum>(r,infini::ops::Sum::DefaultConfig(r.device().type()),contributions->view({tokens,topk,hidden}),std::optional<std::vector<int64_t>>{{1}},false,std::optional<infini::ops::DataType>{},sum);
 ex::submit<infini::ops::Cast>(r,infini::ops::Cast::DefaultConfig(r.device().type()),sum,out);r.syncStream();return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}

extern "C" int modern_grouped_mxfp4_owned(infinilm::runtime::Runtime*owner,size_t tokens,size_t experts,size_t hidden,size_t intermediate,size_t topk,void*out,const void*x,const void*ids,const void*weights,const void*w13,const void*s13,const void*w2,const void*s2){
 auto previous=supplied_runtime;try{owner->assertOwnerThread();if(owner->isGraphRecording())throw std::logic_error("Grouped prefill is eager only");supplied_runtime=owner->shared_from_this();int status=modern_grouped_mxfp4(2,owner->device().index(),tokens,experts,hidden,intermediate,topk,out,x,ids,weights,w13,s13,w2,s2,owner->stream());supplied_runtime=previous;return status;}catch(const std::exception&e){supplied_runtime=previous;error=e.what();return 1;}}

extern "C" int modern_grouped_plan_info(size_t*out){for(size_t i=0;i<5;++i)out[i]=plan_info[i];return 0;}
extern "C" int modern_grouped_trim(int device){try{auto owner=worker(device);owner->assertOwnerThread();owner->syncStream();owner->trimMemory();owner->syncStream();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
extern "C" int modern_grouped_shutdown(){try{for(auto&item:worker_owners()){infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,item.first,true};item.second->syncStream();item.second.reset();}worker_owners().clear();return 0;}catch(const std::exception&e){error=e.what();return 1;}}
