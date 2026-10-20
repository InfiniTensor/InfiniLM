// Test-only bridge: borrowed buffers are held by the synchronous Python caller.
extern "C" int modern_kimi_guard_contract_r3755(int device,int ambient,int T,int R,int P,void**p){
 int previous=-1;cudaGetDevice(&previous);
 try{
  if(!p||device<0||device>=8||ambient<0||ambient>=8)throw std::invalid_argument("3755 contract metadata");
  auto owner=rt::Runtime::create(rt::Device(rt::Device::Type::kHygon,device));auto&r=*owner;r.enableOwnerDrainFailStop();
  {
   auto bf=rt::DataType::kBFloat16,fp=rt::DataType::kFloat32,i32=rt::DataType::kInt32;
   auto wrap=[&](int i,rt::Shape shape,rt::DataType type){return rt::Tensor::from_blob(p[i],shape,type,r.device());};
   auto out=wrap(0,{1,size_t(T),12,128},bf),state=wrap(1,{size_t(P),12,128,128},bf);
   auto q=wrap(2,{1,size_t(T),12,128},bf),k=wrap(3,{1,size_t(T),12,128},bf),v=wrap(4,{1,size_t(T),12,128},bf),g=wrap(5,{1,size_t(T),12,128},bf),b=wrap(6,{1,size_t(T),12},bf);
   auto a=wrap(7,{12},fp),bias=wrap(8,{12,128},fp),off=wrap(9,{size_t(R+1)},i32),read=wrap(10,{size_t(R)},i32),write=wrap(11,{size_t(R)},i32);
   if(cudaSetDevice(ambient)!=cudaSuccess)throw std::runtime_error("3755 select ambient");
   {SpecMathScopeR3676 scope(r,true);spec_kda_roundstep_r3676(r,out,state,q,k,v,g,b,a,bias,off,read,write,1.0f/std::sqrt(128.0f),-5.0f,true);}
   int after=-1;if(cudaGetDevice(&after)!=cudaSuccess||after!=ambient)throw std::runtime_error("3755 failed ambient restoration");r.syncStream();
  }
  r.syncStream();r.trimMemory();owner.reset();if(cudaSetDevice(previous)!=cudaSuccess)throw std::runtime_error("3755 prior device restore");return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"GUARD_CONTRACT_R3755 %s\n",e.what());cudaSetDevice(previous);return 1;}
}
