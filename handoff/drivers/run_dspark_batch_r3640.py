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
 torch.set_num_threads(1);torch.cuda.set_device(device);cpus=sorted(os.sched_getaffinity(0));mask=cpus[device*8:(device+1)*8];assert len(mask)==8;os.sched_setaffinity(0,set(mask));torch.backends.cuda.matmul.allow_tf32=False;baseline_allocated=torch.cuda.memory_allocated(device);block=[2,2,4,4,7,7,7,7][device];path=r/'results'/f'dspark-batch-r3640-{tag}-device{device}.json';assert not path.exists();q={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'node':socket.gethostname(),'device':device,'CPU_affinity':mask,'block_size':block,'additional_proposals_per_round':block,'sample_from_anchor':True,'query_slots':block,'verify_width_if_all_proposed':block+1,'actual_target_verification_executed':False,'state_commit_or_rollback_implemented':False,'end_to_end_15SLO_qualified':False,'cases':[],'source_run':featureproof['source_run'],'feature_audit_sha256':digest(r/'results/kimi-dspark-features-r2773-audit.json'),'protocol_correction':'DSpark sample_from_anchor=True uses allK query outputs; previous2775 DFlash 1+N slice skipped position0 andisnotvalidDSparkacceptance evidence','contract_sources':['sglang/srt/speculative/dspark_components/dspark_config.py get_dspark_sample_from_anchor defaultsTrue','sglang/srt/speculative/dspark_components/dspark_draft.py DraftBlockProposer preservesallgamma hiddenrows','vllm/v1/worker/gpu/spec_decode/dspark/speculator.py sample_from_anchor defaultsTrue'],'scope':'Actual4.5GBpinned DSparkweights forward onreal5layerfull512nativeInfiniLMfeatures. Initialblock only, eager noncausal dual-sourceattention withanchor+MASK, sequentialMarkov/confidenceheads, actualtargetembedding/LMhead. Report offlinefirstblock match againstqualifiednativeB1 sequence, not live targetverify/accept/stateundo orendtoendtok/s. Allrepeat timings recomputefull512context, not cache-amortized steadydecode. 2uniqueprompts,8GPUreplicas,lengths2/4/7, notrepeat-independent dataset.'}
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
  @torch.inference_mode()
  def run_case(case):
   chunks=[]
   for item in case['feature_files']:
    p=r/'results'/item['file'];assert digest(p)==item['sha256'];a=np.fromfile(p,dtype=np.uint16).reshape(1,512,7168).copy();chunks.append(torch.from_numpy(a).view(torch.bfloat16))
   features=torch.cat(chunks,dim=-1).to(device=torch.device('cuda',device)).expand(batch,-1,-1).contiguous();anchor=case['first_token'];assert anchor not in featureproof['eos_token_ids'];mask_id=cfg.dflash_config['mask_token_id']
   with safe_open(target/mapping[embedding_key],framework='pt',device='cpu')as f:
    table=f.get_slice(embedding_key);emb_anchor=table[anchor:anchor+1,:];emb_mask=table[mask_id:mask_id+1,:];assert emb_anchor.shape==emb_mask.shape==(1,7168)
   noise=torch.cat([emb_anchor]+[emb_mask]*(block-1),dim=0).unsqueeze(0).to(device=torch.device('cuda',device)).expand(batch,-1,-1).contiguous();positions=torch.arange(512+block,device=device).reshape(1,-1).expand(batch,-1);reference=None;records=[]
   for repeat in range(4):
    torch.cuda.synchronize(device);a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);a.record();tick=time.perf_counter();hidden=model(position_ids=positions,noise_embedding=noise,target_hidden=features,use_cache=False,is_causal=False);base_logits=F.linear(hidden,head);previous_id=torch.full((batch,1),anchor,device=device,dtype=torch.long);proposals=[];full_logits=[];confidences=[]
    for j in range(block):
     logits=model.markov_head.apply_block_logits(base_logits[:,j:j+1,:],token_ids=previous_id);predicted=logits.argmax(-1);latent=model.markov_head.get_prev_embeddings(previous_id);confidence=torch.sigmoid(model.confidence_head(torch.cat([hidden[:,j:j+1,:],latent],dim=-1)).float());proposals.append(predicted);full_logits.append(logits);confidences.append(confidence);previous_id=predicted
    b.record();b.synchronize();host_ms=(time.perf_counter()-tick)*1000;gpu_ms=a.elapsed_time(b);assert hidden.shape==(batch,block,7168)and torch.isfinite(hidden).all();predicted_ids=torch.cat(proposals,dim=1).cpu().reshape(batch,block).tolist();confidence_values=torch.cat(confidences,dim=1).cpu().reshape(batch,block).tolist();raw_hidden=hidden.cpu().contiguous().view(torch.uint16).numpy().tobytes();all_logits=torch.cat(full_logits,dim=1).cpu().contiguous();assert torch.isfinite(all_logits).all();raw_logits=all_logits.view(torch.uint16).numpy().tobytes();value=(predicted_ids,hashlib.sha256(raw_hidden).hexdigest(),hashlib.sha256(raw_logits).hexdigest());assert reference is None or value[0]==reference[0],('Real_DSpark_token_repeat',device,case['label'],repeat);raw_repeat=reference is None or value[1:]==reference[1:];reference=value;match=[]
    for row in predicted_ids:
     matched=0
     for actual,expected in zip(row,case['target_greedy_reference_ids_first16'][1:]):
      if actual!=expected or actual in featureproof['eos_token_ids']:break
      matched+=1
     match.append(matched)
    stem=f'dspark-batch-r3640-{tag}-device{device}-B{batch}-K{block}-{case["label"]}-r{repeat}';files={}
    for name,data in[('hidden',raw_hidden),('Markov_logits',raw_logits)]:
     p=r/'results'/(stem+'-'+name+'.bin');files[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'persisted':repeat==1};
     if repeat==1:p.write_bytes(data);files[name]['file']=p.name
    records.append({'repeat':repeat,'additional_proposed_ids':predicted_ids,'known_anchor_id':anchor,'native_reference_next_ids':case['target_greedy_reference_ids_first16'][1:block+1],'contiguous_additional_greedy_matches':match,'anchor_is_already_known_not_counted_as_accepted_proposal':True,'confidence_probabilities':confidence_values,'GPU_completed_interval_ms':gpu_ms,'host_completed_ms':host_ms,'cold_first_round':repeat==0,'same_configuration_full_token_repeat_exact':True,'raw_repeat_diagnostic':raw_repeat,'batch':batch,'independent_CPUargmax_checked':torch.equal(all_logits.float().argmax(-1).reshape(batch,block),torch.tensor(predicted_ids)),'files':files})
   return {'label':case['label'],'input_tokens':512,'feature_source_phase':case['feature_phase'],'batch':batch,'block_size':block,'records':records}
  selected=[x for x in featureproof['cases']if x['repeat']==1];assert len(selected)==2
  q['scope']='Actual pinned DSpark fullweights batched firstblock from frozen3e0 native target features, B1/2/4/8/16,K2/4/7. Target feature compatibility tolatest06d and onlineintegration remain unqualified. Alltimings recompute512token draftcontext, no amortizedcache or endtoendtargetspeedup claim; firstblockmatch isoffline match only, not liveacceptance.';q['one_load_many_tests']=True
  for batch in [1,2,4,8,16]:
   for block in [2,4,7]:
    for case in selected:q['cases'].append(run_case(case));save()
  assert sample_weights(model)==before_weights;q['sampled_all62_weights_unchanged']=True;weak_tensors=[weakref.ref(x)for x in list(model.parameters())+list(model.buffers())+[head]];weak_model=weakref.ref(model);del model,head;gc.collect();torch.cuda.synchronize(device);torch.cuda.empty_cache();assert weak_model()is None and all(x()is None for x in weak_tensors);q['Torch_allocated_bytes_before_model']=baseline_allocated;q['Torch_allocated_bytes_after_close']=torch.cuda.memory_allocated(device);q['all_model_parameters_buffers_head_weakrefs_released']=True;q['cleanup_scope']='Allownedmodelparameter/buffer/head Pythonweakrefs expired andstreamcomplete. TorchBLAS internalworkspace mayremain untilnormalprocess exit; noallocatorzero orlive-servicecache-recycleclaim.';q['normal_Torch_tensor_cleanup']=True;q['completed']=True;save()
 except BaseException as e:q['error']=repr(e);save();raise
if __name__=='__main__':
 stage=json.loads((r/'results/dspark-reference-r2764-stage.json').read_text());tiny=json.loads((r/'results/dspark-reference-r2764-tiny.json').read_text());feature=json.loads((r/'results/kimi-dspark-features-r2773-audit.json').read_text());assert stage['ready']and tiny['completed']and feature['completed']
 for file,h in stage['files_sha256'].items():assert digest(base/file)==h
 h=hashlib.sha256()
 with(source/'model.safetensors').open('rb')as f:
  while data:=f.read(16*1024**2):h.update(data)
 assert h.hexdigest()=='ecd746459b4a603ce0d2c64f73935efead29bd651b14439b99e57ee8b41b77ca';assert torch.cuda.device_count()==8 and socket.gethostname()=='COMPUTE_NODE_77b8d291';mp.spawn(worker,args=(feature,),nprocs=8,join=True);print('DSPARK_BATCH_R3640_ALL8_CLOSED',flush=True)
