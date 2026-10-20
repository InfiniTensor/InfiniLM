extern "C" int modern_glm_model_expert_order_policy_r3348(void*raw,int enabled){
 auto*c=static_cast<Context*>(raw);bool owned=false;
 try{if(!c||!c->ready||(enabled!=0&&enabled!=1))throw std::invalid_argument("3348 context/policy");auto&r=*c->owner;r.assertOwnerThread();owned=true;
  if(r.isGraphRecording()||model::decode_leases_r1354::active(r)||!c->module->all_sequences_empty(r))throw std::logic_error("3348 requires reset idle owner outside graph recording");
  r.syncStream();model::expert_order_r3348::policy(r,enabled!=0);return 0;
 }catch(const std::exception&e){error=e.what();if(c&&owned)drain(c);return 1;}
}
extern "C" int modern_glm_model_expert_order_stats_r3348(void*raw,uint64_t*out,size_t count){
 auto*c=static_cast<Context*>(raw);try{if(!c||!c->ready||!out||count!=6)throw std::invalid_argument("3348 stats context/output/count");auto values=model::expert_order_r3348::stats(*c->owner);std::copy(values.begin(),values.end(),out);return 0;
 }catch(const std::exception&e){error=e.what();return 1;}
}
extern "C" size_t modern_glm_expert_order_contexts_r3348(){return model::expert_order_r3348::states.size();}
