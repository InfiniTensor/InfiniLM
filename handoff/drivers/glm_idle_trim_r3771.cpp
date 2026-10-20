// Explicit idle-owner operation; default serving behavior is unchanged.
extern "C" int modern_glm_idle_trim_r3771(void*raw,uint64_t*out,size_t count){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{
  if(!c||!c->ready||!out||count!=6)throw std::invalid_argument("3771 trim context/output/count");
  auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||r.graphPinnedMemoryBytes()!=0||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("3771 trim requires reset eager idle owner without graph pins");
  infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};r.syncStream();size_t before=0,after=0,total=0,total_after=0;
  if(cudaMemGetInfo(&before,&total)!=cudaSuccess)throw std::runtime_error("3771 before cudaMemGetInfo");
  r.trimMemory();r.syncStream();
  if(cudaMemGetInfo(&after,&total_after)!=cudaSuccess||total!=total_after)throw std::runtime_error("3771 after cudaMemGetInfo");
  const uint64_t values[]={before,after,total,uint64_t(r.device().index()),uint64_t(c->stage),uint64_t(c->pp)};std::memcpy(out,values,sizeof(values));return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}
