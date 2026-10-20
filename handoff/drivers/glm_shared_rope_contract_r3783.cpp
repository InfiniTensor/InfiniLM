#include "glm_shared_rope_r3783.hpp"
#include <nlohmann/json.hpp>
#include <thread>
#include <fstream>
namespace sr=infinilm::models::glm_moe_dsa::shared_rope_r3783;namespace rt=infinilm::runtime;using json=nlohmann::json;
extern "C" int glm_shared_rope_contract_r3783(const char*output){std::array<json,8>reports;std::array<std::exception_ptr,8>errors{};std::vector<std::thread>threads;for(int d=0;d<8;++d)threads.emplace_back([&,d]{try{
 auto a=rt::Runtime::create({rt::Device::Type::kHygon,d}),b=rt::Runtime::create({rt::Device::Type::kHygon,d});a->enableOwnerDrainFailStop();b->enableOwnerDrainFailStop();std::weak_ptr<rt::Runtime>wa=a,wb=b;std::weak_ptr<infinilm::layers::RoPE>wx,wy;
 {auto x=sr::get(*a,64,64,1048576,8000000.),xx=sr::get(*a,64,64,1048576,8000000.),y=sr::get(*a,128,64,1048576,8000000.),z=sr::get(*b,64,64,1048576,8000000.);wx=x;wy=y;if(x!=xx||x==y||x==z||x->max_seq_len()!=1048576||y->head_dim()!=128)throw std::runtime_error("3783 sharing/owner key mismatch");if(sr::stats(*a)!=std::array<uint64_t,5>{3,2,1,2,2}||sr::stats(*b)!=std::array<uint64_t,5>{1,1,0,1,1})throw std::runtime_error("3783 actual cache counters");a->syncStream();b->syncStream();}
 if(!wx.expired()||!wy.expired()||sr::stats(*a)[4]!=0||sr::stats(*b)[4]!=0)throw std::runtime_error("3783 pool retained live RoPE");a->trimMemory();b->trimMemory();a.reset();b.reset();if(!wa.expired()||!wb.expired())throw std::runtime_error("3783 Runtime cycle");reports[d]={{"device",d},{"same_Runtime_head_key_shares",true},{"different_head_and_Runtime_isolated",true},{"original_1M_context_preserved",true},{"weakpool_does_not_retain_RoPE_or_Runtime",true},{"both_normal_owners_closed",true}};
 }catch(...){errors[d]=std::current_exception();}});for(auto&t:threads)t.join();json q={{"completed",true},{"owners",reports},{"scope","Exact sharedfactory usedbyModel, actual latestCommon Hygon full1M RoPE constructors, sharing/ownerisolation/counters/weaklifetime. No modeltoken/performancequalification."}};int code=0;for(auto&e:errors)if(e){try{std::rethrow_exception(e);}catch(const std::exception&x){q["error"]=x.what();}q["completed"]=false;code=1;break;}std::ofstream out(output);out<<q.dump(2);return code;
}
