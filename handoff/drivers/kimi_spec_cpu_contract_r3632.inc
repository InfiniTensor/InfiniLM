#include <functional>
extern "C" int modern_kimi_spec_contract_r3632(){try{
 using namespace infinilm::models::kimi_native;
 size_t positive=0,negative=0;
 auto reject=[&](const std::function<void()>&f){bool caught=false;try{f();}catch(const std::invalid_argument&){caught=true;}if(!caught)throw std::runtime_error("3632 invalid CPU contract accepted");++negative;};
 if(!spec_enabled_r3632())throw std::invalid_argument("3632 explicit test optin required");
 for(size_t B: {size_t(1),size_t(2),size_t(4),size_t(8),size_t(16),size_t(32),size_t(64)})for(size_t W=1;W<=8;++W){
  size_t cap=2*B,N=B*W;std::vector<size_t>lengths(cap+1,0);std::vector<int32_t>rows(B),read(B),write(B),past(B),total(B),offset(B+1),cu(B+1),table(B*4);std::vector<int64_t>ids(N,17),positions(N),slots(N);
  for(size_t i=0;i<B;++i){rows[i]=read[i]=int32_t(i+1);write[i]=int32_t(i+1+B);past[i]=int32_t(7+i);lengths[i+1]=size_t(past[i]);total[i]=past[i]+int32_t(W);offset[i+1]=int32_t((i+1)*W);cu[i+1]=cu[i]+total[i];int32_t page=1+int32_t(i)*4;for(size_t j=0;j<4;++j)table[i*4+j]=page+int32_t(j);for(size_t j=0;j<W;++j){positions[i*W+j]=past[i]+int64_t(j);slots[i*W+j]=(page+positions[i*W+j]/256)*256+positions[i*W+j]%256;}}
  en::InferEngine::Input a;a.input_ids=cpu(ids.data(),{1,N},rt::DataType::kInt64);a.position_ids=cpu(positions.data(),{1,N},rt::DataType::kInt64);a.past_sequence_lengths=cpu(past.data(),{B},rt::DataType::kInt32);a.total_sequence_lengths=cpu(total.data(),{B},rt::DataType::kInt32);a.input_offsets=cpu(offset.data(),{B+1},rt::DataType::kInt32);a.cu_seqlens=cpu(cu.data(),{B+1},rt::DataType::kInt32);a.block_tables=cpu(table.data(),{B,4},rt::DataType::kInt32);a.slot_mapping=cpu(slots.data(),{N},rt::DataType::kInt64);a.mamba_init_state_indices=cpu(read.data(),{B},rt::DataType::kInt32);a.mamba_final_state_indices=cpu(write.data(),{B},rt::DataType::kInt32);a.sample_all_positions=true;a.top_k=1;a.top_p=1;a.temperature=1;
  int owner=1;
  {SpecScopeR3632 scope(&owner,cap,rows,read,write);auto b=a;auto plan=freeze_input(b,lengths,163840,&owner);if(plan.size()!=B)throw std::runtime_error("3632 plan size");auto committed=lengths;commit(committed,plan);for(size_t i=0;i<B;++i)if(plan[i].row!=size_t(write[i])||committed[size_t(read[i])]!=lengths[size_t(read[i])]||committed[size_t(write[i])]!=size_t(total[i]))throw std::runtime_error("3632 shadow length commit");++positive;}
  if(B==2&&W==2){
   reject([&]{auto b=a;freeze_input(b,lengths,163840,&owner);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,read);});
   reject([&]{auto bad=write;std::swap(bad[0],bad[1]);SpecScopeR3632 s(&owner,cap,rows,read,bad);});
   reject([&]{auto bad=rows;bad[1]=bad[0];SpecScopeR3632 s(&owner,cap,bad,read,write);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);SpecScopeR3632 t(&owner,cap,rows,read,write);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;freeze_input(b,lengths,163840,nullptr);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;b.sample_all_positions=false;freeze_input(b,lengths,163840,&owner);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto bad=lengths;bad[1]++;auto b=a;freeze_input(b,bad,163840,&owner);});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;table[0]++;try{freeze_input(b,lengths,163840,&owner);}catch(...){table[0]--;throw;}table[0]--;});
   reject([&]{SpecScopeR3632 s(&owner,cap,rows,read,write);auto b=a;positions[0]++;try{freeze_input(b,lengths,163840,&owner);}catch(...){positions[0]--;throw;}positions[0]--;});
  }
 }
 if(active_spec_proof_r3632||positive!=56||negative!=10)throw std::runtime_error("3632 CPU test counts/TLS cleanup");
 serialized=nlohmann::json{{"completed",true},{"valid_metadata_cases",positive},{"invalid_metadata_cases",negative},{"batch_sizes",{1,2,4,8,16,32,64}},{"widths",{1,2,3,4,5,6,7,8}},{"GPU_math_tested",false}}.dump();return 0;
 }catch(const std::exception&e){error=e.what();return 1;}}
extern "C" const char*modern_kimi_spec_contract_report_r3632(){return serialized.c_str();}
