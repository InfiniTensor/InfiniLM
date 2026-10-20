// Policy changes are allowed only on an idle owner after model reset.
extern "C" int modern_glm_prefill_capacity_policy_r3816(void*raw,int enabled){
 auto*c=static_cast<Context*>(raw);try{
  if(!c||!c->ready||enabled<0||enabled>1)throw std::invalid_argument("3816 policy arguments");
  auto&r=*c->owner;r.assertOwnerThread();
  if(r.isGraphRecording()||r.graphPinnedMemoryBytes()!=0||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("3816 policy requires reset eager idle owner");
  model::glm_prefill_capacity_policy_r3816(r,enabled);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" int modern_glm_prefill_capacity_stats_r3816(void*raw,uint64_t*out,size_t count){
 auto*c=static_cast<Context*>(raw);try{
  if(!c||!c->ready)throw std::invalid_argument("3816 stats context");
  model::glm_prefill_capacity_stats_r3816(*c->owner,out,count);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
