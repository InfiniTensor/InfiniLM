import datetime,hashlib,json,pathlib,shutil
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';parent=r/'candidates/kimi-reduce-3e0d89fe-r3226/Common/InfiniLM';target=r/'candidates/kimi-wide-3e0d89fe-r3393/Common/InfiniLM';out=r/'results/kimi-wide-r3393-stage.json';assert not target.exists()and not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
proof=json.loads((r/'results/kimi-reduce-common-r3227-build.json').read_text());assert proof['completed']and sha(parent/'python/infinilm/_lib/libinfinilm_runtime.so')==proof['runtime_sha256'];cast=json.loads((r/'results/cast64-contract-r3382-audit.json').read_text());assert cast['completed']and cast['qualified']
shutil.copytree(parent,target,ignore=shutil.ignore_patterns('.git','.xmake','build','dist','__pycache__','*.so','*.o','*.a','InfiniCore','native-checks-*'));before={p.relative_to(target).as_posix():sha(p)for p in(target/'csrc').rglob('*')if p.is_file()};p=target/'csrc/engine/distributed/communication_group.cpp';s=p.read_text();a=s.index('    if(input->dtype()==infinilm::runtime::DataType::kBFloat16&&op==infinicclSum&&reduce_precision_r3226::enabled(runtime)){');b=s.index('    const auto dtype = cclDataType(input->dtype());',a)
s=s[:a]+'''    const auto precision_r3393=reduce_precision_r3226::selected(runtime);
    const bool original_f32_r3393=input->dtype()==infinilm::runtime::DataType::kFloat32;
    if(op==infinicclSum&&((precision_r3393==1&&input->dtype()==infinilm::runtime::DataType::kBFloat16)||(precision_r3393==2&&(original_f32_r3393||input->dtype()==infinilm::runtime::DataType::kBFloat16)))){
        infinilm::runtime::Tensor wide;
        try{
            wide=infinilm::runtime::Tensor::empty(runtime,input->shape(),precision_r3393==2?infinilm::runtime::DataType::kFloat64:infinilm::runtime::DataType::kFloat32);
            infinilm::execution::submit<infini::ops::Cast>(runtime,infini::ops::Cast::DefaultConfig(runtime.device().type()),input,wide);
            allreduce_(runtime,wide,wide,op,communicator);
            infinilm::execution::submit<infini::ops::Cast>(runtime,infini::ops::Cast::DefaultConfig(runtime.device().type()),wide,output);
            runtime.syncStream(); // Scratch survives the collective and narrowing cast.
            reduce_precision_r3226::record(runtime,wide->nbytes());
            if(precision_r3393==2)reduce_precision_r3226::record_wide(runtime,original_f32_r3393,wide->nbytes());
            INFINILM_TRACE_OP(AllReduce,output);return;
        }catch(...){try{runtime.syncStream();}catch(...){std::terminate();}throw;}
    }
'''+s[b:];p.write_text(s)
for src,dst in [('kimi_wide_precision_r3393.hpp','kimi_reduce_precision_r3226.hpp'),('kimi_wide_precision_r3393.inc','kimi_reduce_precision_r3226.inc')]:shutil.copy2(r/'incoming'/src,p.parent/dst)
changed=[n for n,h in before.items()if sha(target/n)!=h];assert set(changed)=={'csrc/engine/distributed/communication_group.cpp','csrc/engine/distributed/kimi_reduce_precision_r3226.hpp','csrc/engine/distributed/kimi_reduce_precision_r3226.inc'}
out.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ready':True,'pins':proof['pins'],'parent_runtime_sha256':proof['runtime_sha256'],'changed':changed,'csrc_sha256':{'Common':{p.relative_to(target).as_posix():sha(p)for p in(target/'csrc').rglob('*')if p.is_file()}},'cast_qualification_sha256':sha(r/'results/cast64-contract-r3382-audit.json'),'scope':'PrivateCommon3226 sourcecopy only. CAPI3226 ABI retained, policy0baseline/1BF16->FP32/2BF16andFP32->FP64 SUM eager actualmultirequest; F64recursion bypass, B1unpromoted, owner/nested/busy guards, sync-heldscratch and6wide counters. KimiModel3261privateGroupABI reusedunchanged; nofullmodelqualification. ComponentF64CCL audit required beforebuild/use.'},indent=2));print(json.dumps({'ready':True,'changed':changed}))
