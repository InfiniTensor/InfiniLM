#pragma once
#include "runtime/runtime.hpp"
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
extern "C" int deep_trace_hash_r390(const void*,void*,size_t,size_t,size_t,void*);
namespace infinilm::models::deepseek_v4 {
inline thread_local std::map<const void*,size_t> swa_layers_r609;
inline void swa_trace_r609(runtime::Runtime&r,const char*point,const runtime::Tensor&x,size_t per_request=1){
 const char*tag=std::getenv("DEEP_DUP_R591_TAG"),*width=std::getenv("DEEP_DUP_R591_WIDTH");if(!tag||!width||std::string(width)!="1")return;
 size_t n=std::stoull(std::getenv("DEEP_DUP_R591_REQUESTS"));
 if(!x||x->dtype()!=runtime::DataType::kBFloat16||!x->is_contiguous()||!per_request||x->numel()%(n*per_request))throw std::runtime_error("609 BF16 trace geometry differs");
 size_t columns=x->numel()/n/per_request;auto hashes=runtime::Tensor::empty(r,{n,128},runtime::DataType::kInt64);
 if(deep_trace_hash_r390(x->data(),hashes->data(),n,per_request,columns,r.stream()))throw std::runtime_error("609 hash failed");
 std::vector<uint64_t>host(n*128);r.memcpyD2H(host.data(),hashes->data(),host.size()*8);bool same=true;for(size_t i=1;i<n;++i)for(size_t j=0;j<128;++j)same&=host[i*128+j]==host[j];
 std::string stem="/refactor/results/deep-attention-r609-"+std::string(std::getenv("SLURM_JOB_ID"))+"."+std::string(std::getenv("SLURM_STEP_ID"))+"-rank"+std::to_string(r.device().index())+"-"+tag;
 nlohmann::json row={{"point",point},{"shape",x->shape()},{"requests",n},{"tokens_per_request",per_request},{"width",columns},{"hash_lanes",host},{"duplicate_rows_hash_exact",same}};
 if(r.device().index()==0||r.device().index()==6){std::vector<char>raw(x->nbytes());r.memcpyD2H(raw.data(),x->data(),raw.size());std::string file=stem+"-"+point+".bin";std::ofstream out(file,std::ios::binary);out.write(raw.data(),raw.size());out.close();if(!out)throw std::runtime_error("609 raw snapshot failed");row["raw_file"]=file;row["raw_bytes"]=raw.size();}
 std::ofstream log(stem+".jsonl",std::ios::app);log<<row.dump()<<'\n';if(!log)throw std::runtime_error("609 trace journal failed");
}
}
