import pathlib,ast
r=pathlib.Path(__file__).resolve().parent;s=(r/'kimi_projection_replay_r3866.cpp').read_text().replace('r3866','r3875');a=s.index('  b=rt::Tensor::from_blob',s.index('extern "C" int kimi_projection_r3875_gemm'));b=s.index('  r.syncStream();return 0;',a)
s=s[:a]+'''  b=rt::Tensor::from_blob(const_cast<void*>(w),{7168,1536},rt::DataType::kBFloat16,r.device());
  out=rt::Tensor::from_blob(y,{m,7168},rt::DataType::kBFloat16,r.device());
  if(packed==0){
    infinilm::layers::linear::dense_projection(r,a,b,std::optional<rt::Tensor>{},1.f,std::nullopt,false,out);
  }else if(packed==1){
    auto aa=rt::Tensor::from_blob(const_cast<void*>(x),{m,1,1536},rt::DataType::kBFloat16,r.device());
    auto bb=rt::Tensor::from_blob(const_cast<void*>(w),{1,7168,1536},rt::DataType::kBFloat16,r.device())->permute({0,2,1});
    auto yy=rt::Tensor::from_blob(y,{m,1,7168},rt::DataType::kBFloat16,r.device());
    infinilm::execution::submit<infini::ops::Gemm>(r,infinilm::execution::RecordingPolicy::DeviceOnly,infini::ops::Gemm::DefaultConfig(r.device().type()),aa,bb,std::optional<rt::Tensor>{},std::optional<float>{1.f},std::optional<float>{0.f},std::optional<int>{},std::optional<int>{},yy);
  }else{
    for(size_t i=0;i<m;++i){
      auto aa=rt::Tensor::from_blob(const_cast<char*>(static_cast<const char*>(x))+i*1536*2,{1,1536},rt::DataType::kBFloat16,r.device());
      auto yy=rt::Tensor::from_blob(static_cast<char*>(y)+i*7168*2,{1,7168},rt::DataType::kBFloat16,r.device());
      infinilm::layers::linear::dense_projection(r,aa,b,std::optional<rt::Tensor>{},1.f,std::nullopt,false,yy);
    }
  }
'''+s[b:];s=s.replace('(packed!=0&&packed!=1)','(packed<0||packed>2)');(r/'kimi_projection_replay_r3875.cpp').write_text(s,encoding='utf8',newline='\n')
s=(r/'kimi_projection_replay_r3866_inside.sh').read_text().replace('r3866','r3875');(r/'kimi_projection_replay_r3875_inside.sh').write_text(s,encoding='utf8',newline='\n')
s=(r/'probe_kimi_projection_replay_r3866.py').read_text().replace('r3866','r3875');a=s.index('   for mode in[0,1,2]:');b=s.index('   assert raw(x)==xb',a)
s=s[:a]+'''   serial=[]
   for row in range(m):
    yy=torch.empty((1,7168),device=rank,dtype=torch.bfloat16)
    ck(l.kimi_projection_r3875_gemm(ctx,1,0,x[row:row+1].data_ptr(),w.data_ptr(),yy.data_ptr(),stream.cuda_stream));serial.append(yy)
   golden=torch.cat(serial);gb=raw(golden);c['serial_M1_reference']=record(label+'-M1-reference',golden);c['projection_variants']=[]
   for variant in[0,1,2]:
    yy=torch.full_like(local,float('nan'));tick=time.perf_counter()
    ck(l.kimi_projection_r3875_gemm(ctx,m,variant,x.data_ptr(),w.data_ptr(),yy.data_ptr(),stream.cuda_stream));stream.synchronize()
    times=[]
    for loop in range(8):
     begin=time.perf_counter();ck(l.kimi_projection_r3875_gemm(ctx,m,variant,x.data_ptr(),w.data_ptr(),yy.data_ptr(),stream.cuda_stream));stream.synchronize();times.append(time.perf_counter()-begin)
    assert bool(torch.isfinite(yy).all())
    c['projection_variants'].append({'variant':variant,'name':['ordinary_2D','broadcast_batched_M1','sequential_M1'][variant],'raw_equal_serial_M1_diagnostic':raw(yy)==gb,'different_elements':int((yy.view(torch.int16)!=golden.view(torch.int16)).sum().item()),'wall_seconds_host_inclusive':times,'output':record(label+'-variant'+str(variant),yy)})
'''+s[b:];s=s.replace("q['collected']=True","q['collected']=True;q['scope']='Actual KDA1 inputs/weights, local GEMM shape component only; no collective or wholemodel token/performance qualification.'");ast.parse(s);(r/'probe_kimi_projection_replay_r3875.py').write_text(s,encoding='utf8',newline='\n')
s=(r/'launch_KDA1_projection_replay_r3866.py').read_text().replace('r3866','r3875');(r/'launch_KDA1_projection_replay_r3875.py').write_text(s,encoding='utf8',newline='\n');print('Three projection shapes, exact captured real inputs; component only.')
