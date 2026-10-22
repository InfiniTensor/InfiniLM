"""Full pinned draft checkpoint: GPU/cache contract, no online acceptance claim."""
import os
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import pathlib,json,hashlib,time,datetime,statistics,gc,weakref
import numpy as np
import torch
from safetensors import safe_open
from transformers import DynamicCache
from modal_reference_r4258 import DFlashDraftModel,Qwen3Config
r=pathlib.Path('/refactor');out=r/'results/modal-full-draft-r4487-GPU.json';assert not out.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();verified=json.loads((r/'results/Modal-draft-r4255-data-verified.json').read_text());contract=json.loads((r/'incoming/modal_reference_contract_r4258.json').read_text());assert verified['completed']and verified['revision']=='c192d15a43407bf758b5ae0880d5c72052fef1de';assert sha(r/'incoming/modal_reference_r4258.py')==contract['module_sha256'];torch.set_num_threads(2);torch.cuda.set_device(0);torch.manual_seed(4480);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
d={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'producer':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'cases':[],'trained_online_speculative_speedup_qualified':False,'target_acceptance_measured':False,'scope':'Full pinned Modal BF16 weights plus actual target embedding/head, one GPU/one load. Synthetic six-tap feature tensors exercise W8/W16 forward, cache crop, advancing-prefix vs fresh recompute and repeated greedy draft IDs; no target verifier/actual acceptance/user tok/s claim. Host-complete draft-only cost includes head/argmax/copy and is not serving speedup. No GPU F64.'}
def save():
 p=out.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(out)
save();model=None;embed=None;head=None
try:
 torch.cuda.synchronize();d['initial_allocated_bytes']=torch.cuda.memory_allocated();cfg=Qwen3Config(**contract['config']);cfg._attn_implementation='sdpa'
 with torch.device('meta'):model=DFlashDraftModel(cfg)
 assert{k:list(v.shape)for k,v in model.state_dict().items()}=={k:v['shape']for k,v in contract['tensor_metadata'].items()};start=time.perf_counter();source=r/'models/Kimi-K3-Modal-DFlash-c192d15a-r4255/model.safetensors';assert source.stat().st_size==verified['bytes'];state={}
 with safe_open(source,framework='pt',device='cpu')as f:
  for name in f.keys():state[name]=f.get_tensor(name)
 model.load_state_dict(state,strict=True,assign=True);model.rotary_emb=type(model.rotary_emb)(cfg);assert not any(v.is_meta for v in model.buffers());model.eval().to(device='cuda');del state;torch.cuda.synchronize();d['single_backbone_weight_load_seconds']=time.perf_counter()-start;d['backbone_parameters']=sum(p.numel()for p in model.parameters());d['backbone_all_BF16']=all(p.dtype==torch.bfloat16 for p in model.parameters());assert d['backbone_all_BF16'];d['rotary_buffer_dtypes']={name:str(v.dtype)for name,v in model.rotary_emb.named_buffers()};assert all(v.dtype==torch.float32 for v in model.rotary_emb.buffers())
 target=pathlib.Path('/shared/Kimi-K3');mapping=json.loads((target/'model.safetensors.index.json').read_text())['weight_map'];start=time.perf_counter();d['target_auxiliary_weights']={}
 def target_weight(key):
  with safe_open(target/mapping[key],framework='pt',device='cpu')as f:w=f.get_tensor(key).contiguous()
  assert w.shape==(163840,7168)and w.dtype==torch.bfloat16;d['target_auxiliary_weights'][key]={'shape':list(w.shape),'sha256':hashlib.sha256(w.view(torch.uint16).numpy()).hexdigest()};return w.to('cuda')
 embed=target_weight('language_model.model.embed_tokens.weight');head=target_weight('language_model.lm_head.weight');torch.cuda.synchronize();d['single_target_embedding_head_load_seconds']=time.perf_counter()-start;d['resident_allocated_bytes']=torch.cuda.memory_allocated();save()
 projection_control=0
 class Projector(torch.nn.Module):
  def __init__(self,original):
   super().__init__();self.original=original;assert original.bias is None;self.register_buffer('float_weight',original.weight.float())
  def forward(self,x):
   if projection_control==0:return self.original(x)
   if projection_control==1:return torch.nn.functional.linear(x.float(),self.float_weight).to(x.dtype)
   assert projection_control==2
   if not x.numel():return self.original(x)
   flat=x.reshape(-1,x.shape[-1]);parts=[]
   for begin in range(0,flat.shape[0],16):
    rows=flat[begin:begin+16];n=rows.shape[0]
    if n<16:rows=torch.nn.functional.pad(rows,(0,0,0,16-n))
    parts.append(self.original(rows)[:n])
   return torch.cat(parts,dim=0).reshape(*x.shape[:-1],self.original.weight.shape[0])
 model.fc=Projector(model.fc)
 for layer in model.layers:
  layer.self_attn.k_proj=Projector(layer.self_attn.k_proj);layer.self_attn.v_proj=Projector(layer.self_attn.v_proj)
 d['projection_controls']='Native/F32-output-thenBF16/fixedM16 BF16 onfc+alllayerK/V, no source/operator/targetchanges. Same synthetic inputs percontrol; comparecache semantics within eachmode, notdraftIDcrossmodeoronlineacceptance. FP32 permitted/noGPUF64. ExtraFP32weightbuffers retained acrossallmodes to keepresidentmemory constant.'
 with torch.inference_mode():
  for projection_label,projection_control in [('native-before',0),('F32',1),('fixedM16',2),('native-after',0)]:
   torch.cuda.synchronize();torch.manual_seed(4487)
   for B in [1,2]:
    for W in [8,16]:
     for prefix in [32,512]:
      features=torch.randn(B,prefix,6*7168,device='cuda',dtype=torch.bfloat16)*.1;ids=torch.full((B,W),163592,device='cuda',dtype=torch.long);ids[:,0]=41677;noise=torch.nn.functional.embedding(ids,embed);cache=DynamicCache();positions=torch.arange(prefix+W,device='cuda').view(1,-1).expand(B,-1)
      def forward(position,feature,past=None):
       hidden=model(position_ids=position,noise_embedding=noise,target_hidden=feature,past_key_values=past,use_cache=past is not None);logits=hidden@head.T;assert torch.isfinite(logits).all();tokens=logits[:,1:,:].argmax(-1).cpu().tolist();torch.cuda.synchronize();return hidden,tokens
      full,reference=forward(positions,features);initial,got=forward(positions,features,cache);assert int(cache.get_seq_length())==prefix+W;equal_initial=got==reference;cache.crop(prefix);empty=features[:,:0,:];samples=[];repeat_ids=[]
      for rep in range(5):
       tick=time.perf_counter();cached,tokens=forward(positions[:,-W:],empty,cache);samples.append((time.perf_counter()-tick)*1000);repeat_ids.append(tokens);assert int(cache.get_seq_length())==prefix+W;cache.crop(prefix)
      row={'projection_control':projection_label,'batch':B,'width':W,'proposals_excluding_anchor':W-1,'prefix':prefix,'initial_cache_vs_recompute_draft_IDs_exact':equal_initial,'cached_vs_recompute_draft_IDs_exact':repeat_ids[0]==reference,'own_repeated_draft_IDs_exact':all(x==repeat_ids[0]for x in repeat_ids),'cached_draft_host_complete_ms_samples':samples,'cached_draft_host_complete_ms_median':statistics.median(samples),'max_hidden_abs_diff_cached_vs_recompute':float((cached-full).abs().max()),'advancing_prefix':[]}
      total_features=features;past=prefix
      for committed in [1,min(4,W-1),W-1]:
       new_features=torch.randn(B,committed,6*7168,device='cuda',dtype=torch.bfloat16)*.1;total_features=torch.cat([total_features,new_features],dim=1);new_past=past+committed;new_positions=torch.arange(past,new_past+W,device='cuda').view(1,-1).expand(B,-1);fresh_positions=torch.arange(new_past+W,device='cuda').view(1,-1).expand(B,-1);advanced,advanced_ids=forward(new_positions,new_features,cache);assert int(cache.get_seq_length())==new_past+W;cache.crop(new_past);fresh,fresh_ids=forward(fresh_positions,total_features);row['advancing_prefix'].append({'synthetic_committed_tokens':committed,'cached_prefix':new_past,'draft_IDs_exact':advanced_ids==fresh_ids,'max_hidden_abs_diff':float((advanced-fresh).abs().max())});past=new_past
      weak=weakref.ref(cache);cache.crop(0);del cache;gc.collect();assert weak()is None;row['cache_released']=True;d['cases'].append(row);save();print('MODAL_FULL_DRAFT_R4487_CASE',B,W,prefix,row['cached_vs_recompute_draft_IDs_exact'],row['own_repeated_draft_IDs_exact'],round(row['cached_draft_host_complete_ms_median'],3),flush=True);del features,total_features,noise,ids,full,initial,cached,advanced,fresh,new_features,empty
 d['per_projection_control']={name:{'cases':len([x for x in d['cases']if x['projection_control']==name]),'all_fixed_prefix_IDs_exact':all(x['cached_vs_recompute_draft_IDs_exact']for x in d['cases']if x['projection_control']==name),'all_advancing_prefix_IDs_exact':all(z['draft_IDs_exact']for x in d['cases']if x['projection_control']==name for z in x['advancing_prefix'])}for name in ['native-before','F32','fixedM16','native-after']}
 d['component_all_draft_ID_contracts_pass']=all(x['initial_cache_vs_recompute_draft_IDs_exact']and x['cached_vs_recompute_draft_IDs_exact']and x['own_repeated_draft_IDs_exact']and all(z['draft_IDs_exact']for z in x['advancing_prefix'])for x in d['cases']);d['qualification_scope']='GPU/cache component only, mismatches retained; no online target acceptance/15tok/s/serving qualification.';d['completed']=True
except BaseException as e:d['error']=repr(e);save();raise
finally:
 model=None;embed=None;head=None;gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();d['GPU_cleanup_synchronized']=True;d['final_allocated_bytes']=torch.cuda.memory_allocated();save()
print('MODAL_FULL_DRAFT_R4487_COMPLETE',d['component_all_draft_ID_contracts_pass'],flush=True)
