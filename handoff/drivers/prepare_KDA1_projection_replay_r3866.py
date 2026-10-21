import pathlib,ast
r=pathlib.Path(__file__).resolve().parent
s=(r/'kimi_projection_replay_r3409.cpp').read_text().replace('r3409','r3866')
s=s.replace('#include "execution/submit.hpp"','#include "execution/submit.hpp"\n#include "layers/linear/dense.hpp"')
a=s.index('std::shared_ptr<rt::Runtime> worker(');b=s.index('struct Context',a)
s=s[:a]+'std::shared_ptr<rt::Runtime> worker(int d){auto r=rt::Runtime::create({rt::Device::Type::kHygon,d});r->enableOwnerDrainFailStop();return r;}\n'+s[b:]
s=s.replace('(m!=4&&m!=8&&m!=16)','(m!=1&&m!=2)')
a=s.index('  infinilm::execution::submit<infini::ops::Gemm>');b=s.index('  r.syncStream();return 0;',a)
s=s[:a]+'''  // Exact NoneQuantization dense_projection path, original unpacked weight.
  auto original=b->permute({1,0});
  infinilm::layers::linear::dense_projection(r,a,original,std::optional<rt::Tensor>{},1.f,std::nullopt,false,out);
'''+s[b:]
s=s.replace('if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))','infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};if(cudaStreamSynchronize(static_cast<cudaStream_t>(stream))') if False else s
# allreduce also owns its device explicitly; no reliance on Runtime's ambient device.
s=s.replace('auto &r=*c->runtime;\n auto dt=', 'auto &r=*c->runtime;infini::rt::DeviceGuard guard{r.device().type(),r.device().index(),true};\n auto dt=')
(r/'kimi_projection_replay_r3866.cpp').write_text(s,encoding='utf8',newline='\n')
s=(r/'kimi_projection_replay_r3409_inside.sh').read_text().replace('kimi-wide-replay-3e0d89fe-r3409','kimi-KDA1-replay-5f91ad7-r3866').replace('kimi-reduce-3e0d89fe-r3226/Model/InfiniLM','kimi-MLA-trace-5f91ad7-r3841/InfiniLM').replace('kimi-wide-3e0d89fe-r3393/Common/InfiniLM','rebased-5f91ad7-r3668/Common/Kimi/InfiniLM').replace('hygon-kimi-cast64-r3315','hygon-kimi-5f91ad7-r3672').replace('r3409','r3866')
s=s.replace('latest-sdk-r2844-Kimi','latest-sdk-r3672-Kimi').replace('kimi-wide-common-r3395','common-latest-r3673-Kimi')
s=s.replace("'scope':'Freshwrapper actualSDK2648/Common2649/CCL6f, notolderprobe binary. Buildonly; realrankinputs numeric/perf unqualified.'", "'scope':'Current5f91 actualSDK3672/Common3673, KDA1 captured real inputs and original checkpoint projection. Build only; component replay is not wholemodel qualification.'")
(r/'kimi_projection_replay_r3866_inside.sh').write_text(s,encoding='utf8',newline='\n')
print('Current dependency replay wrapper prepared; M1/M2 only; no model library replacement.')
