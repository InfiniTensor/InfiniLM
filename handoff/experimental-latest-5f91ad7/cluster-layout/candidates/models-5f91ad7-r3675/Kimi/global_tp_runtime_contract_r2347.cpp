#include "engine/distributed/communication_group.hpp"
#include "runtime/runtime.hpp"
#include <infini/rt/device_guard.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <vector>
namespace rt=infinilm::runtime;namespace en=infinilm::engine::distributed;
using json=nlohmann::json;
void require(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
uint16_t bf16(int n){float f=float(n);uint32_t u;std::memcpy(&u,&f,4);return uint16_t(u>>16);}
extern "C" int modern_global_tp_contract_r2347(int argc,char**argv){if(argc!=7)return 2;json result={{"completed",false},{"scope","1261 actual private CommunicationGroup local8/globalTP16-or32/PP mapping and native Runtime model submission boundary. Collective sums/gathers, PP payload/feedback and normalowner-first cleanup. No model math/loading/15SLO or RDMA qualification."}};
 try{
    const int node=std::stoi(argv[1]),nodes=std::stoi(argv[2]),pp=std::stoi(argv[3]);
    require(node>=0&&node<nodes*pp,"1261 node");
    en::DistConfig cfg(8);cfg.tp_node_size=nodes;cfg.tp_node_rank=node%nodes;cfg.pp_size=pp;cfg.pp_stage=node/nodes;cfg.master_addr=argv[4];cfg.master_port=std::stoi(argv[5]);
    result["node"]=node;result["TP"]=8*nodes;result["PP"]=pp;result["owners"]=json::array();
    {
    en::CommunicationGroup group(cfg,rt::Device::Type::kHygon);
    require(group.tp_size()==8,"1261 worker count must remain local8");
    std::array<json,8>reports;std::array<std::exception_ptr,8>errors{};std::vector<std::thread>threads;
    for(int device=0;device<8;++device)threads.emplace_back([&,device]{
        json report={{"device",device},{"cases",json::array()}};
        try{
            auto rank=group.get_rank_info(device);require(rank.tp_size==8*nodes&&rank.tp_rank==(node%nodes)*8+device&&rank.world_rank==node*8+device&&rank.world_size==8*nodes*pp,"1261 global/local rank fields");
            require(rank.device.index()==device&&rank.pp_stage==node/nodes&&rank.pp_size==pp&&rank.comm,"1261 local device and communicator");
            report["tp_rank"]=rank.tp_rank;report["world_rank"]=rank.world_rank;report["pp_stage"]=rank.pp_stage;
            infini::rt::DeviceGuard guard{rt::Device::Type::kHygon,device,true};
            auto owner=rt::Runtime::create(rank.device);auto&r=*owner;r.enableOwnerDrainFailStop();
            {
            auto x=rt::Tensor::empty(r,{32},rt::DataType::kBFloat16),y=rt::Tensor::empty(r,{32},rt::DataType::kBFloat16);
            std::vector<uint16_t>source(32),actual(32);
            for(int phase=0;phase<3;++phase){
                const bool world=phase==2;auto comm=world&&pp>1?rank.world_comm:rank.comm;
                const int count=world?rank.world_size:rank.tp_size;
                const int value=phase==1?rank.pp_stage+1:1;
                source.assign(32,bf16(value));r.memcpyH2D(x->data(),source.data(),x->nbytes(),false);
                for(int repeat=0;repeat<3;++repeat){en::allreduce_(r,y,x,infinicclSum,comm);r.syncStream();r.memcpyD2H(actual.data(),y->data(),y->nbytes());require(actual==std::vector<uint16_t>(32,bf16(count*value)),"1261 BF16 exact group sum");}
                report["cases"].push_back({{"phase",phase},{"repeats",3},{"exact_group_sum",true}});
            }
            for(bool world:{false,true}){
                const int count=world?rank.world_size:rank.tp_size;auto comm=world&&pp>1?rank.world_comm:rank.comm;
                auto gathered=rt::Tensor::empty(r,{size_t(32*count)},rt::DataType::kBFloat16);
                source.assign(32,bf16(world?rank.world_rank:rank.tp_rank));r.memcpyH2D(x->data(),source.data(),x->nbytes(),false);
                en::allgather_(r,gathered,x,comm);r.syncStream();std::vector<uint16_t>got(32*count);r.memcpyD2H(got.data(),gathered->data(),gathered->nbytes());
                for(int rank_id=0;rank_id<count;++rank_id)for(int j=0;j<32;++j)require(got[rank_id*32+j]==bf16(rank_id),"1261 ordered global allgather");
                report["cases"].push_back({{"world_gather",world},{"ranks",count},{"all_rank_payloads_exact",true}});
            }
            if(pp>1){
                source.assign(32,bf16(rank.tp_rank+1));r.memcpyH2D(x->data(),source.data(),x->nbytes(),false);
                if(rank.pp_stage>0){en::recv_(r,y,rank.world_rank-rank.tp_size,rank.world_comm);r.syncStream();r.memcpyD2H(actual.data(),y->data(),y->nbytes());require(actual==source,"1261 PP predecessor payload");}
                if(rank.pp_stage+1<pp)en::send(r,x,rank.world_rank+rank.tp_size,rank.world_comm);
                r.syncStream();report["PP_neighbor_payload_exact"]=true;
                if(rank.tp_rank==0){
                    if(rank.pp_stage==pp-1){source.assign(32,bf16(42));r.memcpyH2D(x->data(),source.data(),x->nbytes(),false);en::send(r,x,0,rank.world_comm);}
                    if(rank.pp_stage==0){en::recv_(r,y,(pp-1)*rank.tp_size,rank.world_comm);r.syncStream();r.memcpyD2H(actual.data(),y->data(),y->nbytes());require(actual==std::vector<uint16_t>(32,bf16(42)),"1261 greedy feedback target");}
                    r.syncStream();report["tail_feedback_target_exact"]=true;
                }
            }
            }
            r.syncStream();r.trimMemory();owner.reset();report["owner_shutdown_completed"]=true;
        }catch(...){errors[device]=std::current_exception();try{std::rethrow_exception(errors[device]);}catch(const std::exception&e){report["error"]=e.what();}}
        reports[device]=report;
    });
    for(auto&t:threads)t.join();result["owners"]=reports;
    for(auto&e:errors)if(e)std::rethrow_exception(e);
    result["all8_Runtime_owners_closed_before_communicators"]=true;
    }
    result["communication_group_destroyed"]=true;result["completed"]=true;
 }catch(const std::exception&e){result["error"]=e.what();}
 std::ofstream out(argv[6]);out<<result.dump(2);std::cout<<result.dump()<<std::endl;
 return result["completed"].get<bool>()?0:1;
}
