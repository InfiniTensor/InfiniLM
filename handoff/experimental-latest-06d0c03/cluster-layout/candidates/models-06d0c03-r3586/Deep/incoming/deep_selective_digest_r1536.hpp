#pragma once
#include "kimi_gpu_tensor_detail_r1392.hpp"
#include <thread>
namespace infinilm::diagnostics::deep_selective_r1536 {
namespace dg=infinilm::diagnostics::kimi_tensor_detail_r1392;
class Frame;
inline thread_local Frame*current=nullptr;
class Frame {
 runtime::Runtime&owner_;int mode_,selected_;size_t logical_=0;bool active_=true;
 std::thread::id thread_;dg::Guard guard_;
 static runtime::Runtime&validate(runtime::Runtime&r,int mode,int selected){
  r.assertOwnerThread();if(current)throw std::logic_error("1536 nested frame");
  if(mode<0||mode>2||selected< -1||selected>=48)throw std::invalid_argument("1536 mode/selected point");
  if(r.isGraphRecording())throw std::logic_error("1536 diagnostics forbid graphs");return r;
 }
 void check(runtime::Runtime&r)const{
  r.assertOwnerThread();if(&r!=&owner_||std::this_thread::get_id()!=thread_||!active_||current!=this)throw std::logic_error("1536 foreign/closed frame");
 }
public:
 Frame(runtime::Runtime&r,int mode,int tp,int pp,int selected):owner_(validate(r,mode,selected)),mode_(mode),selected_(selected),thread_(std::this_thread::get_id()),guard_(r,mode,tp,pp){current=this;}
 Frame(const Frame&)=delete;Frame&operator=(const Frame&)=delete;
 ~Frame()noexcept{if(active_){if(current!=this||std::this_thread::get_id()!=thread_)std::terminate();current=nullptr;}}
 size_t points()const{return logical_;}
 void mark(runtime::Runtime&r,const runtime::Tensor&value,size_t layer,const char*kind){
  check(r);if(logical_>=48)throw std::invalid_argument("1536 too many logical points");
  const bool record=mode_==2&&(selected_== -1||size_t(selected_)==logical_);
  struct Restore{int value;~Restore(){dg::frame.mode=value;}}restore{dg::frame.mode};
  if(dg::frame.owner&&!record)dg::frame.mode=1;
  dg::mark(r,value,layer,kind);
  if(dg::frame.owner){auto&p=dg::frame.points.back();p["logical_index"]=logical_;p["GPU_hash_recorded"]=record;}
  ++logical_;
 }
 void finish(){
  check(owner_);if(logical_!=48)throw std::logic_error("1536 incomplete frame");guard_.finish();auto value=dg::latest(owner_.device().index());
  value["device"]=owner_.device().index();value["logical_points"]=logical_;value["selected_point"]=selected_;
  value["actual_GPU_hash_launches"]=mode_==2?(selected_== -1?48:1):0;value["all48_GPU_hashes_recorded"]=mode_==2&&selected_== -1;
  dg::publish(owner_,std::move(value));active_=false;current=nullptr;
 }
};
inline void mark(runtime::Runtime&r,const runtime::Tensor&value,size_t layer,const char*kind){if(current)current->mark(r,value,layer,kind);}
inline auto latest(int device){return dg::latest(device);}
}
