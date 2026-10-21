#pragma once
#include "runtime/runtime.hpp"
#include <stdexcept>
extern "C" const char*glm_fp8_error_r776();
extern "C" void*glm_fp8_create_r776(int,int,size_t,size_t,size_t,size_t,size_t,size_t,size_t,void*);
extern "C" int glm_fp8_run_r776(void*,int,void*,const void*,const void*,const void*,const void*);
extern "C" int glm_fp8_destroy_r776(void*);
extern "C" void modern_glm_fp8_count_r780(int);
namespace infinilm::models::glm_moe_dsa {
inline bool fp8_prefill_r780(runtime::Runtime&r,const runtime::Tensor&out,const runtime::Tensor&x,const runtime::Tensor&xs,const runtime::Tensor&w,const runtime::Tensor&ws,size_t ro,size_t co){
 r.assertOwnerThread();if(x->size(0)<32)return false;
 if(r.isGraphRecording()||r.device().type()!=runtime::Device::Type::kHygon||out->dtype()!=runtime::DataType::kBFloat16||x->dtype()!=runtime::DataType::kUInt8||w->dtype()!=x->dtype()||xs->dtype()!=runtime::DataType::kFloat32||ws->dtype()!=xs->dtype()||out->device()!=r.device()||x->device()!=r.device()||xs->device()!=r.device()||w->device()!=r.device()||ws->device()!=r.device()||!out->is_contiguous()||!x->is_contiguous()||!xs->is_contiguous()||!w->is_contiguous()||!ws->is_contiguous()||out->ndim()!=2||x->ndim()!=2||w->ndim()!=2||xs->ndim()!=2||ws->ndim()!=2||out->size(0)!=x->size(0)||out->size(1)!=w->size(0)||x->size(1)!=w->size(1)||xs->size(0)!=x->size(0)||xs->size(1)!=ws->size(1))throw std::invalid_argument("780 owned contiguous FP8 geometry differs");
 void*p=glm_fp8_create_r776(r.device().index(),1,x->size(0),w->size(0),x->size(1),ws->size(0),ws->size(1),ro,co,r.stream());
 if(!p)throw std::runtime_error(glm_fp8_error_r776());
 int status=glm_fp8_run_r776(p,1,const_cast<void*>(out->data()),x->data(),xs->data(),w->data(),ws->data());
 std::string error=status?glm_fp8_error_r776():"";
 int destroyed=glm_fp8_destroy_r776(p);
 if(status)throw std::runtime_error(error);
 if(destroyed)throw std::runtime_error(glm_fp8_error_r776());
 modern_glm_fp8_count_r780(r.device().index());return true;
}
}
