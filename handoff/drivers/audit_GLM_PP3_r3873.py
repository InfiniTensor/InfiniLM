import os,datetime, hashlib, json, math, pathlib, re, subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';mode='shca';ROWS=64;out=r/'results'/'glm-PP3-r3873-audit.json';assert not out.exists()
result={'cache_rows':ROWS,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,
        'scope':'Latest5f91 GLM TP8PP3 G32/64 M8/16 fixed640ctx. All24 owner/Slurm terminal, full token IDs against frozen3330, QA/EOS/repeats/raw/dispatch/SHAs audited. 512input/max128; not HTTP or peak.'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
frozen_profile={8:True,16:True};frozen_differences=[]
def frozen_compare(c,gold,kind,stage,rank):
    exact=all(v==gold[i%8]for i,v in enumerate(c['generated_ids']));frozen_profile[c['micro']] &=exact
    if not exact and stage==2 and rank==0:
        mismatches=[]
        for i,v in enumerate(c['generated_ids']):
            g=gold[i%8];j=next((j for j,(a,b)in enumerate(zip(v,g))if a!=b),min(len(v),len(g))if len(v)!=len(g)else None)
            if j is not None:mismatches.append({'row':i,'index':j,'actual':v[j]if j<len(v)else None,'expected':g[j]if j<len(g)else None})
        frozen_differences.append({'phase':c['phase'],'micro':c['micro'],'kind':kind,'first_differences':mismatches[:4]})

try:
    launch=load(r/'glm-PP3-serving-r3871-launch.json');reference_audit=load(r/'results/glm-fastpath-r3330-audit.json');assert reference_audit['completed'];reference=load(r/'results'/f'glm-fastpath-r3323-{reference_audit["producer_state"][0]}-stage3-rank0.json');refs={c['phase']:c for c in reference['multi_batch_cases']};
    assert load(r/'results/glm-PP3-r3854-build.json')['completed'];
    rows=[a.split('|')for a in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%110,State,ExitCode,NodeList%160'],text=True).splitlines()]
    rows=[a for a in rows if a[1]==launch['label']]
    assert len(rows)==1;state=rows[0];result['producer_state']=state
    assert state[2:4]==['COMPLETED','0:0'],state
    step=state[0];
    transport_logs={}
    for node in range(3):
        log=(r/'logs'/f'{launch["label"]}-{step}-{node}.log').read_text();expected='IBext_v8'if mode=='shca'else'Socket';assert 'Using network '+expected in log;assert 'Using network '+('Socket'if mode=='shca'else'IBext_v8')not in log;assert 'NET/IB : Using [0]shca_'in log if mode=='shca'else 'NET/Socket : Using [0]ib'in log;transport_logs[str(node)]={'sha256':hashlib.sha256(log.encode()).hexdigest(),'actual_backend':expected,'GPU_Direct_RDMA_disabled_logged':'GPU Direct RDMA Disabled'in log}
    reports=[];cpu_raw=0;head=None
    natural=load(r/'results'/f'glm-serving-cases-r2130-{step}.json');eos=set(natural['eos_token_ids'])
    sdk=load(r/'results/latest-sdk-r3672-GLM-build.json');build=load(r/'results/glm-PP3-r3854-build.json')
    common=load(r/'results/common-latest-r3673-GLM-build.json')
    all_refs={};summaries=[]
    for stage in range(3):
        for rank in range(8):
            p=r/'results'/f'glm-fastpath-r3871-{step}-stage{stage}-rank{rank}.json';q=load(p)
            assert q['completed']and q['worker_shutdown_completed']and q['cohort_phases_completed']and q['cache_rows']==ROWS and q['cache_capacity']==640
            assert q['library_sha256']==build['library_sha256']and q['actual_sdk_sha256']==sdk['library_sha256']
            assert q['LM_source_commit']=='5f91ad738583db8f5617aeae53b76523f85d1ec6'
            assert q['batch_cache_default_OFF_verified']and q['batch_cache_wrong_thread_and_invalid_untouched']and q['batch_cache_live_sequence_policy_rejected']
            assert len(q['actual_FP8_helpers'])==2
            for name,h in q['actual_FP8_helpers'].items():
                assert '/candidates/glm-PP3-5f91ad7-r3854/helpers/'in name
                assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h==build['fresh_helpers']['helpers/'+pathlib.Path(name).name]

            for name,h in q['probe_source_sha256'].items():assert sha(r/'incoming'/name)==h
            for name,h in q['actual_LM_native_libraries'].items():
                assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h
                if 'libinfinilm_runtime.so'in name:assert h==common['runtime_sha256']
            assert q['probe_preimport_before_CCL_verified']and q['Python_bytecode_cache_prefix'].startswith('/local/infini-pycache-')
            assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4,q['owner_affinity']
            assert q['BLAS_policy']['failures']==0 and all(v['created']==v['verified']for v in q['BLAS_policy']['owners'])
            after=q['cache_after_destroy'];assert after['created']==after['destroyed']and after['active']==after['active_bytes']==0
            assert q['fastpath_invalid_and_dependency_guards_verified']and q['fastpath_live_sequence_mutation_rejected']
            cfg=load(pathlib.Path.home()/'hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json');bounds=[0,26,54,78];layers=bounds[stage+1]-bounds[stage];full=sum(v=='full'for v in cfg['indexer_types'][bounds[stage]:bounds[stage+1]])
            assert q['fastpath_stage_layers']==layers and q['fastpath_full_indexer_layers']==full
            assert[q['actual_partition']['begin'],q['actual_partition']['end']]==bounds[stage:stage+2]and q['actual_partition']['PP']==3
            for epoch in ['before_select','after_select']:
                assert q[epoch+'_chat_gate_passed']
                gold=load(r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{stage}-rank{rank}.json')[epoch+'_chat_cases']
                for a,b in zip(q[epoch+'_chat_cases'],gold):
                    assert a['generated_ids']==b['generated_ids']
            phases=q['multi_batch_cases'];assert len(phases)==12+2*len([(b,m)for m in[8,16]for b in[32,64]if b<=ROWS])
            # Compare all ranks' output IDs and EOS decisions, not unsynchronized clocks.
            signatures=[(c['phase'],c['generated_ids'],c['stopped_at_EOS'])for c in phases]
            if head is None:head=signatures
            assert signatures==head
            for c in phases:
                if c['qa']:
                    key=re.sub(r'-b(?:4|8|16)(?=-p)', '', c['phase']);gold=refs[key]['generated_ids'];frozen_compare(c,gold,'QA',stage,rank)
                else:
                    gold=refs['g32m8_A1']['generated_ids'];frozen_compare(c,gold,'natural',stage,rank)
                batch=c['batch'];micro=c['micro'];gen=c['generated_ids'];times=c['step_completion_seconds'];limit=c['max_output_tokens']
                assert c['all_serving_cards']==24 and len(gen)==len(times)==batch
                answer_eos=set(q['chat_template_provenance']['eos_token_ids'])if c['qa']else eos
                assert c['no_forward_after_EOS']and c['same_micro_reference_passed']
                assert c['memory_free_after_phase']>0
                for i,(g,ts)in enumerate(zip(gen,times)):
                    assert 1<=len(g)<=limit and len(g)==len(ts)and all(a<b for a,b in zip(ts,ts[1:]))
                    assert all(v not in answer_eos for v in g[:-1])
                    assert g[-1]in answer_eos or len(g)==limit
                    assert c['stopped_at_EOS'][i]==(g[-1]in answer_eos)
                counts=[0]*batch;feedback=0;per_round={}
                for e in c['events']:
                    ids=e['rows'];t=e['round'];assert ids==sorted(set(ids))
                    assert len({i//micro for i in ids})==1
                    assert len(ids)<=(min(8,micro)if t==0 else micro)
                    assert len(ids)*max(c['input_lengths'][i]for i in ids)<=4096 if t==0 else True
                    for i in ids:assert counts[i]==t and t<len(gen[i]);counts[i]+=1
                    per_round.setdefault(t,[]).extend(ids);feedback+=1
                assert counts==[len(g)for g in gen]
                before,after=c['batch_cache_before'],c['batch_cache_after'];delta=[b-a for a,b in zip(before,after)]
                assert delta==c['batch_cache_delta']and before[0]==after[0]==c['batch_cache_policy']and delta[4]==0
                assert c['batch_cache_policy']==1
                enabled=c['fastpath_bundle_policy'];assert enabled==q['fastpath_phase_flags'][c['phase']]
                dc=sum(e['round']>0 for e in c['events']);dr=sum(len(e['rows'])for e in c['events']if e['round']>0)
                expected=[full*dc,full*dc,layers*dr]if enabled else[0,0,0]
                fd=[b-a for a,b in zip(c['fastpath_counters_before'],c['fastpath_counters_after'])];assert fd==c['fastpath_counter_delta']==expected

                assert c['feedback_calls']==(len(per_round)if c['pipeline_round_feedback']else feedback)
                useful=sum(len(g)-int(g[-1]in answer_eos)for g in gen)
                first=max(ts[0]for ts in times);finish=max(ts[-1]for ts in times)
                dec=sum(sum(t>first and tok not in answer_eos for t,tok in zip(ts,g))for ts,g in zip(times,gen))
                assert useful==c['useful_output_tokens']and dec==c['decode_interval_useful_tokens']
                rate=dec/(finish-first)if finish>first else None
                if rate is not None:assert math.isclose(rate,c['aggregate_useful_decode_tokens_s'],rel_tol=1e-12)
                assert math.isclose(useful/c['elapsed_seconds'],c['including_prefill_useful_tokens_s'],rel_tol=1e-12)
                if stage==2:
                    for i,g in enumerate(gen):
                        # Original cohort position and case family define the reference.
                        family=c['phase'].split('-')[1]if c['qa']else i%2
                        key=(stage,rank,c['qa'],family,micro,i%micro,limit)
                        v=g;old=all_refs.setdefault(key,v);assert old==v,('sameM',key,c['phase'])
                if c['qa']:
                    assert all(c['stopped_at_EOS'])
                    if stage==2 and rank==0:
                        assert len(c['answers'])==batch
                        for a in c['answers']:
                            assert a['thinking_closed']and a['known_answer']
                            text=a['answer'];label=c['phase']
                            known=('北京'in text and('中国'in text or'首都'in text))if'beijing'in label else(bool(re.search(r'(?<!\d)3973(?!\d)',text.replace(',','')))if'arithmetic'in label else'待发货'in text)
                            assert known
                elif stage==2 and rank==0:
                    summaries.append({k:c[k]for k in ['phase','batch','micro','pipeline_round_feedback','feedback_calls','elapsed_seconds','aggregate_useful_decode_tokens_s','decode_tokens_s_per_card','including_prefill_useful_tokens_s','including_prefill_tokens_s_per_card','useful_output_tokens','useful_tokens_per_request','TTFT_seconds_per_request','batch_cache_policy','batch_cache_delta','fastpath_bundle_policy','fastpath_counter_delta']})
                    summaries[-1]['input_tokens_per_second_to_all_prefill_complete']=sum(c['input_lengths'])/first
            if stage==2 and rank==0:
                for c in q['raw_checkpoints']:
                    p=r/'results'/c['file'];assert sha(p)==c['sha256'];raw=p.read_bytes();x=(np.frombuffer(raw,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(c['shape'])
                    assert np.isfinite(x).all()and np.argmax(x,axis=1).tolist()==c['GPU_argmax']
                    phase=next(v for v in phases if v['phase']==c['phase'])
                    for j,i in enumerate(c['rows']):assert hashlib.sha256(raw[j*x.shape[1]*2:(j+1)*x.shape[1]*2]).hexdigest()==phase['raw_hashes_per_request'][i][str(c['step'])]
                    cpu_raw+=1
            reports.append({'stage':stage,'rank':rank,'file':f'glm-fastpath-r3871-{step}-stage{stage}-rank{rank}.json','report_sha256':sha(r/'results'/f'glm-fastpath-r3871-{step}-stage{stage}-rank{rank}.json')})
    repeated=[]
    for batch,micro in [(b,m)for m in[8,16]for b in[32,64]if b<=ROWS]:
        group=[v for v in summaries if v['batch']==batch and v['micro']==micro]
        assert len(group)==2 and all(v['fastpath_bundle_policy']==1 for v in group)
        import statistics
        dec=statistics.median(v['aggregate_useful_decode_tokens_s']for v in group)
        inc=statistics.median(v['including_prefill_useful_tokens_s']for v in group)
        repeated.append({'batch':batch,'micro':micro,'decode_median':dec,'including_prefill_median':inc,'decode_per_all24_cards':dec/24,'including_prefill_per_all24_cards':inc/24,'prefill_input_tok_s_median':statistics.median(v['input_tokens_per_second_to_all_prefill_complete']for v in group),'repetitions':2,'scope':'512input/max128 realEOS, repeated2promptfamilies; forward/reverse config ordering; no sameprocessOFFcontrol, HTTP or hardwaremaximum.'})
    for v in repeated:v['complete_frozen_ID_profile_qualified']=frozen_profile[v['micro']]
    result['diagnostic_all_profile_summary']=repeated
    result['qualified_configuration_summary']=[v for v in repeated if v['complete_frozen_ID_profile_qualified']]
    result['micro_frozen_reference_qualification']=frozen_profile
    result['rejected_profile_complete_token_differences']=frozen_differences
    result['configuration_admission_policy']='A micro geometry qualifies only if every QA and natural case in all24 reports matches complete frozen3330 IDs incl EOS and ownrepeats; sameengine resetbeforeeachphase verified by producer. Othergeometry realtokenmismatch preserved andexcluded. No referencechange/normalization/raw-basedrejection.'

    result.update(completed=True,normal_workers_closed=24,CPU_fullvocab_raw_files=cpu_raw,
                  all24_IDs_EOS_schedule_verified=True,all_tail_sameM_token_repeats_verified=True,external_frozen3330_complete_IDs_exact_for_all_multibatch=all(frozen_profile.values()),B1_knownQA_complete_IDs_exact_to_frozen999=True,source_pins=build['pins'],logits_bitwise_equality_required=False,transport=mode,actual_transport_logs=transport_logs,
                  performance=summaries,reports=reports)
except BaseException as e:
    import traceback
    result.update(error=repr(e),traceback=traceback.format_exc())
    out.write_text(json.dumps(result,indent=2));raise
out.write_text(json.dumps(result,indent=2));print('GLM_PROFILE_R3797_AUDIT_COMPLETE',ROWS,frozen_profile,cpu_raw,flush=True)
