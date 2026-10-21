import ctypes
import datetime,hashlib,json,os,pathlib,socket,sys,time,gc,weakref
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ.pop('SPECFORGE_DRAFT_FLEX_BACKEND',None)
import numpy as np
import torch,torch.multiprocessing as mp
import torch.nn.functional as F
from safetensors import safe_open
r=pathlib.Path('/refactor');tag=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];source=r/'models/Kimi-K3-DSpark-3c5bac30-r2699';base=r/'candidates/dspark-reference-3c5bac30-r2764';package=json.loads((r/'results/dspark-reference-r2764-stage.json').read_text());assert package['ready']
for file,h in package['files_sha256'].items():assert hashlib.sha256((base/file).read_bytes()).hexdigest()==h
sys.path.insert(0,str(base));from dspark import DSparkConfig,DSparkDraftModel
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sample_weights(model):
 result={}
 for name,p in model.named_parameters():
  view=p.detach().reshape(-1);indices=torch.tensor([0,view.numel()//2,view.numel()-1],device=p.device);result[name]=view[indices].cpu().view(torch.uint16).tolist()
 return result
@torch.inference_mode()
def worker(device,featureproof):
 torch.set_num_threads(1);torch.cuda.set_device(device);cpus=sorted(os.sched_getaffinity(0));mask=cpus[device*8:(device+1)*8];assert len(mask)==8;os.sched_setaffinity(0,set(mask));torch.backends.cuda.matmul.allow_tf32=False;baseline_allocated=torch.cuda.memory_allocated(device);block=[2,2,4,4,7,7,7,7][device];path=r/'results'/f'dspark-fused-r4164-{tag}-device{device}.json';assert not path.exists();q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'node':socket.gethostname(),'device':device,'CPU_affinity':mask,'block_size':block,'additional_proposals_per_round':block,'sample_from_anchor':True,'query_slots':block,'verify_width_if_all_proposed':block+1,'actual_target_verification_executed':False,'state_commit_or_rollback_implemented':False,'end_to_end_15SLO_qualified':False,'cases':[],'source_run':featureproof['source_run'],'feature_audit_sha256':digest(r/'results/kimi-dspark-features-r2773-audit.json'),'protocol_correction':'DSpark sample_from_anchor=True uses allK query outputs; previous2775 DFlash 1+N slice skipped position0 andisnotvalidDSparkacceptance evidence','contract_sources':['sglang/srt/speculative/dspark_components/dspark_config.py get_dspark_sample_from_anchor defaultsTrue','sglang/srt/speculative/dspark_components/dspark_draft.py DraftBlockProposer preservesallgamma hiddenrows','vllm/v1/worker/gpu/spec_decode/dspark/speculator.py sample_from_anchor defaultsTrue'],'scope':'Actual4.5GBpinned DSparkweights forward onreal5layerfull512nativeInfiniLMfeatures. Initialblock only, eager noncausal dual-sourceattention withanchor+MASK, sequentialMarkov/confidenceheads, actualtargetembedding/LMhead. Report offlinefirstblock match againstqualifiednativeB1 sequence, not live targetverify/accept/stateundo orendtoendtok/s. Allrepeat timings recomputefull512context, not cache-amortized steadydecode. 2uniqueprompts,8GPUreplicas,lengths2/4/7, notrepeat-independent dataset.'}
 def save():
  tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(path)
 save();start=time.perf_counter()
 try:
  raw=json.loads((source/'config.json').read_text());cfg=DSparkConfig(**raw);cfg._attn_implementation='eager';assert cfg.dflash_config['target_layer_ids']==[7,23,51,67,83]and cfg.hidden_size==7168 and cfg.num_hidden_layers==5
  previous=torch.get_default_dtype()
  try:torch.set_default_dtype(torch.bfloat16);model=DSparkDraftModel(cfg).eval()
  finally:torch.set_default_dtype(previous)
  with safe_open(source/'model.safetensors',framework='pt',device='cpu')as f:
   state={k:f.get_tensor(k)for k in f.keys()};assert len(state)==62 and all(v.dtype==torch.bfloat16 for v in state.values());assert set(model.state_dict())==set(state);model.load_state_dict(state,strict=True)
  del state;model.to(torch.device('cuda',device));assert model.rotary_emb.rope_type=='yarn'and model.rotary_emb.inv_freq.dtype==torch.float32;assert all(p.dtype==torch.bfloat16 and p.device.index==device and torch.isfinite(p).all()for p in model.parameters());before_weights=sample_weights(model)
  target=pathlib.Path('/shared/Kimi-K3');mapping=json.loads((target/'model.safetensors.index.json').read_text())['weight_map'];head_keys=[k for k in mapping if k.endswith('lm_head.weight')];embedding_keys=[k for k in mapping if k.endswith('embed_tokens.weight')];assert len(head_keys)==len(embedding_keys)==1;head_key=head_keys[0];embedding_key=embedding_keys[0]
  with safe_open(target/mapping[head_key],framework='pt',device='cpu')as f:
   head=f.get_tensor(head_key);assert head.shape==(163840,7168)and head.dtype==torch.bfloat16;head=head.to(device=torch.device('cuda',device))
  q['target_weight_keys']={'embedding':embedding_key,'LMhead':head_key};q['loading_seconds']=time.perf_counter()-start;q['loaded_draft_tensor_count']=62;q['resident_Torch_bytes_after_load']=torch.cuda.memory_allocated(device);save()
  proof=json.loads((r/'results/dspark-markov-r4163-build.json').read_text());assert proof['completed'];fused_path=r/'candidates/dspark-markov-r4163/libdspark_markov_r4163.so';assert digest(fused_path)==proof['library_sha256'];fused_lib=ctypes.CDLL(str(fused_path));fused_fn=fused_lib.dspark_markov_logits_r4163;fused_fn.argtypes=[ctypes.c_int]*4+[ctypes.c_size_t]+[ctypes.c_void_p]*6;fused_fn.restype=ctypes.c_int;q['fused_library_sha256']=proof['library_sha256'];q['fused_source_sha256']=proof['source_sha256']
  @torch.inference_mode()
  def run_case(case):
   from transformers.cache_utils import DynamicCache
   chunks=[]
   for item in case['feature_files']:
    p=r/'results'/item['file'];assert digest(p)==item['sha256'];a=np.fromfile(p,dtype=np.uint16).reshape(1,512,7168).copy();chunks.append(torch.from_numpy(a).view(torch.bfloat16))
   features=torch.cat(chunks,dim=-1).to(device=torch.device('cuda',device)).expand(batch,-1,-1).contiguous();anchor=case['first_token'];assert anchor not in featureproof['eos_token_ids'];mask_id=cfg.dflash_config['mask_token_id']
   with safe_open(target/mapping[embedding_key],framework='pt',device='cpu')as f:
    table=f.get_slice(embedding_key);emb_anchor=table[anchor:anchor+1,:];emb_mask=table[mask_id:mask_id+1,:]
   noise=torch.cat([emb_anchor]+[emb_mask]*(block-1),dim=0).unsqueeze(0).to(device=torch.device('cuda',device)).expand(batch,-1,-1).contiguous()
   full_positions=torch.arange(512+block,device=device).reshape(1,-1).expand(batch,-1)
   query_positions=torch.arange(512,512+block,device=device).reshape(1,-1).expand(batch,-1)
   def heads(hidden):
    base_logits=F.linear(hidden,head);previous_id=torch.full((batch,1),anchor,device=device,dtype=torch.long);proposals=[];logits=[];confidence=[]
    for j in range(block):
     x=model.markov_head.apply_block_logits(base_logits[:,j:j+1,:],token_ids=previous_id);token=x.argmax(-1);latent=model.markov_head.get_prev_embeddings(previous_id);c=torch.sigmoid(model.confidence_head(torch.cat([hidden[:,j:j+1,:],latent],dim=-1)).float());proposals.append(token);logits.append(x);confidence.append(c);previous_id=token
    return torch.cat(proposals,dim=1),torch.cat(logits,dim=1),torch.cat(confidence,dim=1)
   cache=DynamicCache();torch.cuda.synchronize(device);init=time.perf_counter()
   initial=model(position_ids=full_positions,noise_embedding=noise,target_hidden=features,past_key_values=cache,use_cache=True,is_causal=False);torch.cuda.synchronize(device);init_ms=(time.perf_counter()-init)*1000;assert int(cache.get_seq_length())==512+block
   cache.crop(512);assert int(cache.get_seq_length())==512;del initial
   references={};records=[];cache_weak=weakref.ref(cache)
   for cache_enabled,repeat in[(False,0),(True,0),(True,1),(False,1),(False,2),(True,2),(True,3),(False,3)]:
    cache.crop(512);assert int(cache.get_seq_length())==512;torch.cuda.synchronize(device);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();tick=time.perf_counter()
    hidden=model(position_ids=query_positions if cache_enabled else full_positions,noise_embedding=noise,target_hidden=features[:,:0,:]if cache_enabled else features,past_key_values=cache if cache_enabled else None,use_cache=cache_enabled,is_causal=False)
    predicted,logits,confidence=heads(hidden);b.record();b.synchronize();host_ms=(time.perf_counter()-tick)*1000;gpu_ms=a.elapsed_time(b)
    assert hidden.shape==(batch,block,7168)and torch.isfinite(hidden).all()and torch.isfinite(logits).all()and torch.isfinite(confidence).all()
    assert int(cache.get_seq_length())==(512+block if cache_enabled else 512)
    ids=predicted.cpu().tolist();cpu_logits=logits.cpu().contiguous();assert cpu_logits.float().argmax(-1).tolist()==ids
    own=references.setdefault(cache_enabled,ids);assert ids==own,('draft_cache_own_ID_repeat',device,batch,block,cache_enabled)
    raw=cpu_logits.view(torch.uint16).numpy().tobytes();persist=repeat==1 and(batch==1 or(batch==8 and block==16 and cache_enabled));name=f'dspark-fused-r4164-{tag}-d{device}-B{batch}-K{block}-{case["label"]}-c{int(cache_enabled)}-r{repeat}.bin';file={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'persisted':persist}
    if persist:p=r/'results'/name;assert not p.exists();p.write_bytes(raw);file['file']=name
    gold=case['target_greedy_reference_ids_first16'][1:];matched=[]
    for row in ids:
     n=0
     for x,y in zip(row,gold):
      if x!=y or x in featureproof['eos_token_ids']:break
      n+=1
     matched.append(n)
    records.append({'cache_enabled':cache_enabled,'repeat':repeat,'additional_proposed_ids':ids,'known_anchor_id':anchor,'contiguous_greedy_matches_to_old_target':matched,'gold_future_available':len(gold),'acceptance_censored_at_reference_end':[n==len(gold)and block>len(gold)for n in matched],'confidence_probabilities':confidence.cpu().reshape(batch,block).tolist(),'GPU_completed_interval_ms':gpu_ms,'host_completed_ms':host_ms,'cache_length_after_forward':int(cache.get_seq_length()),'CPUargmax_checked':True,'finite':True,'own_ID_repeat_exact':True,'file':file})
    del hidden,predicted,logits,confidence,cpu_logits
   # Additional finite component experiment; old first-block references only.
   phase_records=[]
   for repeat in range(6):
    cache.crop(512);events=[torch.cuda.Event(enable_timing=True)for _ in range(4)];torch.cuda.synchronize(device);tick=time.perf_counter();events[0].record()
    hidden=model(position_ids=query_positions,noise_embedding=noise,target_hidden=features[:,:0,:],past_key_values=cache,use_cache=True,is_causal=False);events[1].record()
    base_logits=F.linear(hidden,head);events[2].record();previous_id=torch.full((batch,1),anchor,device=device,dtype=torch.long);proposals=[];all_logits=[];all_conf=[]
    for j in range(block):
     x=model.markov_head.apply_block_logits(base_logits[:,j:j+1,:],token_ids=previous_id);token=x.argmax(-1);latent=model.markov_head.get_prev_embeddings(previous_id);c=torch.sigmoid(model.confidence_head(torch.cat([hidden[:,j:j+1,:],latent],dim=-1)).float());proposals.append(token);all_logits.append(x);all_conf.append(c);previous_id=token
    predicted=torch.cat(proposals,dim=1);logits=torch.cat(all_logits,dim=1);confidence=torch.cat(all_conf,dim=1);events[3].record();events[-1].synchronize();host_ms=(time.perf_counter()-tick)*1000
    assert predicted.cpu().tolist()==references[True]and logits.float().argmax(-1).cpu().tolist()==references[True]and torch.isfinite(logits).all()and torch.isfinite(confidence).all()
    phase_records.append({'repeat':repeat,'host_ms':host_ms,'backbone_device_interval_ms':events[0].elapsed_time(events[1]),'LMhead_device_interval_ms':events[1].elapsed_time(events[2]),'sequential_Markov_confidence_concat_device_interval_ms':events[2].elapsed_time(events[3]),'CPUargmax':True,'finite':True,'IDs_exact_to_cached_eager':True})
    del hidden,base_logits,predicted,logits,confidence,all_logits,all_conf,proposals,x,c,latent,token,previous_id
   class FixedPrefix:
    def __init__(self):self.prefix={};self.initializing=True
    def update(self,k,v,index,cache_kwargs=None):
     if self.initializing:
      assert k.shape[-2]==512+block and index not in self.prefix;self.prefix[index]=(k[...,:512,:].contiguous().clone(),v[...,:512,:].contiguous().clone());return k,v
     assert k.shape[-2]==v.shape[-2]==block and index in self.prefix
     pk,pv=self.prefix[index];assert pk.device==k.device and pv.device==v.device and pk.dtype==k.dtype and pv.dtype==v.dtype and pk.shape[:-2]==k.shape[:-2]
     return torch.cat([pk,k],dim=-2),torch.cat([pv,v],dim=-2)
   fixed=FixedPrefix();initial=model(position_ids=full_positions,noise_embedding=noise,target_hidden=features,past_key_values=fixed,use_cache=True,is_causal=False);torch.cuda.synchronize(device);assert len(fixed.prefix)==5;del initial;fixed.initializing=False
   prefix_before={i:(a[...,0,:].clone(),b[...,-1,:].clone())for i,(a,b)in fixed.prefix.items()}
   anchor_input=torch.full((batch,1),anchor,device=device,dtype=torch.long);static_noise=noise.clone()
   def fused_markov(base,previous,guards):
    V,R=163840,256;w1=model.markov_head.markov_w1.weight;w2=model.markov_head.markov_w2.weight
    assert base.shape==(batch,1,V)and base.dtype==torch.bfloat16 and base.stride(-1)==1 and previous.shape==(batch,1)and previous.dtype==torch.int64 and previous.is_contiguous()and w1.shape==w2.shape==(V,R)and w1.is_contiguous()and w2.is_contiguous()and w1.dtype==w2.dtype==base.dtype and base.device==previous.device==w1.device==w2.device
    storage=torch.full((batch*V+128,),17.,device=device,dtype=torch.bfloat16);out=storage[64:-64].view(batch,1,V);guards.append(storage)
    code=fused_fn(device,batch,V,R,base.stride(0),previous.data_ptr(),w1.data_ptr(),w2.data_ptr(),base.data_ptr(),out.data_ptr(),torch.cuda.current_stream(device).cuda_stream);assert code==0,('fused_Markov_status',code);return out
   def captured_step(use_fused=True):
    hidden=model(position_ids=query_positions,noise_embedding=static_noise,target_hidden=features[:,:0,:],past_key_values=fixed,use_cache=True,is_causal=False);base=F.linear(hidden,head);previous=anchor_input;pred=[];logits=[];conf=[];guards=[]
    for j in range(block):
     x=fused_markov(base[:,j:j+1,:],previous,guards)if use_fused else model.markov_head.apply_block_logits(base[:,j:j+1,:],token_ids=previous);token=x.argmax(-1);latent=model.markov_head.get_prev_embeddings(previous);c=torch.sigmoid(model.confidence_head(torch.cat([hidden[:,j:j+1,:],latent],dim=-1)).float());pred.append(token);logits.append(x);conf.append(c);previous=token
    return torch.cat(pred,dim=1),torch.cat(logits,dim=1),torch.cat(conf,dim=1),guards
   graph_records=[];graph=None;graph_outputs=None;normal_graph_reset=False
   try:
    eager=captured_step();torch.cuda.synchronize(device);assert eager[0].cpu().tolist()==references[True];del eager
    capture_stream=torch.cuda.Stream(device=device);capture_stream.wait_stream(torch.cuda.current_stream());graph=torch.cuda.CUDAGraph();start_capture=time.perf_counter()
    with torch.cuda.graph(graph,stream=capture_stream,capture_error_mode='relaxed'):graph_outputs=captured_step()
    torch.cuda.current_stream().wait_stream(capture_stream);torch.cuda.synchronize(device);capture_ms=(time.perf_counter()-start_capture)*1000
    for variant in range(2):
     other=case if variant==0 else next(x for x in selected if x['label']!=case['label']);new_anchor=other['first_token']
     with safe_open(target/mapping[embedding_key],framework='pt',device='cpu')as f:new_embedding=f.get_slice(embedding_key)[new_anchor:new_anchor+1,:]
     noise_input=noise.clone();noise_input[:,0,:].copy_(new_embedding.to(device=torch.device('cuda',device)));static_noise.copy_(noise_input);anchor_input.fill_(new_anchor);torch.cuda.synchronize(device);expected=captured_step(False);expected_ids=expected[0].cpu().tolist();assert torch.isfinite(expected[1]).all();del expected
     columns=[0,1,7,127,1023,163839];oracle_base=None
     if batch==1:
      hidden_oracle=model(position_ids=query_positions,noise_embedding=static_noise,target_hidden=features[:,:0,:],past_key_values=fixed,use_cache=True,is_causal=False);base_oracle=F.linear(hidden_oracle,head);oracle_base=base_oracle[0,:,columns].float().cpu().tolist();del hidden_oracle,base_oracle
     for graph_enabled,repeat in[(False,0),(True,0),(True,1),(False,1),(False,2),(True,2),(True,3),(False,3)]:
      a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);torch.cuda.synchronize(device);a.record();tick=time.perf_counter()
      if graph_enabled:graph.replay();values=graph_outputs
      else:values=captured_step()
      b.record();b.synchronize();host_ms=(time.perf_counter()-tick)*1000;gpu_ms=a.elapsed_time(b);ids=values[0].cpu().tolist();cpu_logits=values[1].cpu().contiguous();assert ids==expected_ids and cpu_logits.float().argmax(-1).tolist()==ids and torch.isfinite(values[1]).all()and torch.isfinite(values[2]).all()
      assert len(values[3])==block and all(bool((buf[:64]==17).all()and(buf[-64:]==17).all())for buf in values[3])
      raw=cpu_logits.view(torch.uint16).numpy().tobytes();persist=batch==1 and repeat==1;item={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'persisted':persist}
      if persist:
       name=f'dspark-fused-r4164-{tag}-d{device}-B{batch}-K{block}-{case["label"]}-v{variant}-g{int(graph_enabled)}-r{repeat}.bin';p=r/'results'/name;assert not p.exists();p.write_bytes(raw);item['file']=name
      graph_records.append({'variant':variant,'known_anchor_id':new_anchor,'prefix_still_original_prompt':True,'graph_enabled':graph_enabled,'repeat':repeat,'host_ms':host_ms,'device_ms':gpu_ms,'IDs':ids,'expected_eager_IDs':expected_ids,'IDs_match':True,'CPUargmax':True,'finite':True,'raw':item,'fused_guard_regions':True,'CPU_oracle_sample':{'columns':columns,'base':oracle_base,'previous_ids':[new_anchor]+expected_ids[0][:-1],'actual':cpu_logits[0,:,columns].float().tolist()}if batch==1 and repeat==1 else None});del values,cpu_logits
     del noise_input,new_embedding
   finally:
    torch.cuda.synchronize(device)
    if graph is not None:graph.reset();normal_graph_reset=True
   for i,(a,b)in fixed.prefix.items():assert torch.equal(a[...,0,:],prefix_before[i][0])and torch.equal(b[...,-1,:],prefix_before[i][1])
   del graph,graph_outputs,fixed,prefix_before,static_noise,anchor_input
   graph_result={'capture_ms':capture_ms,'actual_device_graph':True,'normal_reset_before_cache_free':normal_graph_reset,'prefix_sample_readonly':True,'records':graph_records,'scope':'Static512 prefix only. Variant1 changesactualanchor+noise atsameoriginalcontext toprove graphreadsdeviceinputs; not a valid secondtarget continuation oronlineacceptance. No advancingKV/statecommit proof.'}

   cache.crop(512);del cache;gc.collect();assert cache_weak()is None
   return {'label':case['label'],'batch':batch,'block_size':block,'input_tokens':512,'feature_source_phase':case['feature_phase'],'draft_prefix_cache_init_ms':init_ms,'cache_vs_recompute_IDs_exact':references[True]==references[False],'cache_object_released':True,'K16_extrapolation_not_trained_length':block==16,'records':records,'performance_breakdown':phase_records,'graph_experiment':graph_result}
  selected=[x for x in featureproof['cases']if x['repeat']==1];assert len(selected)==2
  q['scope']='Actual pinned DSpark fullweights batched firstblock from frozen3e0 native target features, B1/2/4/8/16,K2/4/7. Target feature compatibility tolatest06d and onlineintegration remain unqualified. Alltimings recompute512token draftcontext, no amortizedcache or endtoendtargetspeedup claim; firstblockmatch isoffline match only, not liveacceptance.';q['one_load_many_tests']=True;q['source_driver_sha256']=digest(pathlib.Path(__file__));q['scope']='Pinned3c5bac DSpark firstblock withold3e0 targetfeatures only, newsteadyquerycache vsfull512recompute ABBA. B1/2/4/8 K4/7/8/16, K16explicitextrapolation. Cacheinitexcluded, allMarkov/headwork included, percasecachecrop512 checked. Notlatesttargetacceptance oruserTPS/modelqualification. Newphasebreakdown and actualstaticprefixdevicegraph withchangedanchors, resetbeforefree. FusedMarkov pureHIP vs originalTorch completeproposalIDs+sampledCPUFP64oracle pendingaudit.'
  for batch in [1,2,4,8]:
   for block in [4,7,8,16]:
    for case in selected:q['cases'].append(run_case(case));save()
  assert sample_weights(model)==before_weights;q['sampled_all62_weights_unchanged']=True;weak_tensors=[weakref.ref(x)for x in list(model.parameters())+list(model.buffers())+[head]];weak_model=weakref.ref(model);del model,head;gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache();assert weak_model()is None and all(x()is None for x in weak_tensors);q['Torch_allocated_bytes_before_model']=baseline_allocated;q['Torch_allocated_bytes_after_close']=torch.cuda.memory_allocated(device);q['all_model_parameters_buffers_head_weakrefs_released']=True;q['cleanup_scope']='Allownedmodelparameter/buffer/head Pythonweakrefs expired andstreamcomplete. TorchBLAS internalworkspace mayremain untilnormalprocess exit; noallocatorzero orlive-servicecache-recycleclaim.';q['normal_Torch_tensor_cleanup']=True;q['completed']=True;save()
 except BaseException as e:q['error']=repr(e);save();raise
if __name__=='__main__':
 stage=json.loads((r/'results/dspark-reference-r2764-stage.json').read_text());tiny=json.loads((r/'results/dspark-reference-r2764-tiny.json').read_text());feature=json.loads((r/'results/kimi-dspark-features-r2773-audit.json').read_text());assert stage['ready']and tiny['completed']and feature['completed']
 for file,h in stage['files_sha256'].items():assert digest(base/file)==h
 h=hashlib.sha256()
 with(source/'model.safetensors').open('rb')as f:
  while data:=f.read(16*1024**2):h.update(data)
 assert h.hexdigest()=='ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca';assert torch.cuda.device_count()==8 and socket.gethostname()=='a02r4n17';mp.spawn(worker,args=(feature,),nprocs=8,join=True);print('DSPARK_FUSED_R4164_ALL8_CLOSED',flush=True)
