import os,datetime, hashlib, json, math, pathlib, re, subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';mode='shca';PP=int(os.environ['GLM_IDLE_AUDIT_PP']);assert PP in[2,4];CARDS=8*PP;out=r/'results'/f'glm-idle-r3773-PP{PP}-audit.json';assert not out.exists()
result={'PP':PP,'all_serving_cards':CARDS,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,
        'scope':'Latest5f91 GLM host-only idletrim PP2vsPP4,rows16cap640; allservingcards,realEOS,QA,completeID,CPUargmax,owner/actualdependencySHA/transport/Slurm; G1M1/G8M8/G16M8/G16M16 sameprocessfastpathABBA. Compare nodecounts descriptively, differentnodes/capacity fromhistoricalbaseline; notHTTP orhardwaremax.'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
try:
    launch=next(j for j in load(r/'glm-idle-serving-r3772-launch.json')['jobs']if j['PP']==PP);reference_audit=load(r/'results/glm-fastpath-r3330-audit.json');assert reference_audit['completed'];reference=load(r/'results'/f'glm-fastpath-r3323-{reference_audit["producer_state"][0]}-stage3-rank0.json');refs={c['phase']:c for c in reference['multi_batch_cases']};
    assert load(r/'results/glm-idle-trim-r3771-build.json')['completed'];
    rows=[a.split('|')for a in subprocess.check_output(['sacct','-j',launch['parent'],'-n','-P','-o','JobID,JobName%110,State,ExitCode,NodeList%160'],text=True).splitlines()]
    rows=[a for a in rows if a[1]==launch['label']]
    assert len(rows)==1;state=rows[0];result['producer_state']=state
    assert state[2:4]==['COMPLETED','0:0'],state
    step=state[0];
    transport_logs={}
    for node in range(PP):
        log=(r/'logs'/f'{launch["label"]}-{step}-{node}.log').read_text();expected='IBext_v8'if mode=='shca'else'Socket';assert 'Using network '+expected in log;assert 'Using network '+('Socket'if mode=='shca'else'IBext_v8')not in log;assert 'NET/IB : Using [0]shca_'in log if mode=='shca'else 'NET/Socket : Using [0]ib'in log;transport_logs[str(node)]={'sha256':hashlib.sha256(log.encode()).hexdigest(),'actual_backend':expected,'GPU_Direct_RDMA_disabled_logged':'GPU Direct RDMA Disabled'in log}
    reports=[];cpu_raw=0;head=None
    natural=load(r/'results'/f'glm-serving-cases-r2130-{step}.json');eos=set(natural['eos_token_ids'])
    sdk=load(r/'results/latest-sdk-r3672-GLM-build.json');build=load(r/'results/glm-idle-trim-r3771-build.json')
    common=load(r/'results/common-latest-r3673-GLM-build.json')
    all_refs={};summaries=[]
    for stage in range(PP):
        for rank in range(8):
            p=r/'results'/f'glm-fastpath-r3772-{step}-stage{stage}-rank{rank}.json';q=load(p)
            assert q['completed']and q['worker_shutdown_completed']and q['cohort_phases_completed']and q['cache_rows']==16 and q['cache_capacity']==640
            assert q['library_sha256']==build['library_sha256']and q['actual_sdk_sha256']==sdk['library_sha256']
            assert q['LM_source_commit']=='5f91ad738583db8f5617aeae53b76523f85d1ec6'
            assert q['batch_cache_default_OFF_verified']and q['batch_cache_wrong_thread_and_invalid_untouched']and q['batch_cache_live_sequence_policy_rejected']
            assert len(q['actual_FP8_helpers'])==2
            for name,h in q['actual_FP8_helpers'].items():
                assert '/candidates/glm-idle-trim-5f91ad7-r3771/helpers/'in name
                assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h==build['fresh_helpers']['helpers/'+pathlib.Path(name).name]

            for name,h in q['probe_source_sha256'].items():assert sha(r/'incoming'/name)==h
            for name,h in q['actual_LM_native_libraries'].items():
                assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h
                if 'libinfinilm_runtime.so'in name:assert h==common['runtime_sha256']
            assert q['probe_preimport_before_CCL_verified']and q['Python_bytecode_cache_prefix'].startswith('/local/infini-pycache-')
            assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4,q['owner_affinity']
            assert q['BLAS_policy']['failures']==0 and all(v['created']==v['verified']for v in q['BLAS_policy']['owners'])
            after=q['cache_after_destroy'];assert after['created']==after['destroyed']and after['active']==after['active_bytes']==0
            trim=q['idle_trim_after_loading'];assert trim['foreignthread_and_invalid_rejected'];z=trim['native_before_after_total_device_stage_PP'];assert z[3:]==[rank,stage,PP]and z[0]>0 and z[1]>0 and z[2]>z[1];assert trim['torch_after_bytes']>6*1024**3;assert len(q['idle_trim_phases'])==40
            assert q['fastpath_invalid_and_dependency_guards_verified']and q['fastpath_live_sequence_mutation_rejected']
            cfg=load(pathlib.Path.home()/'hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json');bounds=[0,38,78]if PP==2 else[0,18,38,58,78];layers=bounds[stage+1]-bounds[stage];full=sum(v=='full'for v in cfg['indexer_types'][bounds[stage]:bounds[stage+1]])
            assert q['fastpath_stage_layers']==layers and q['fastpath_full_indexer_layers']==full
            for epoch in ['before_select','after_select']:
                assert q[epoch+'_chat_gate_passed']
                gold=load(r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{3 if stage==PP-1 else 0}-rank{rank}.json')[epoch+'_chat_cases']
                for a,b in zip(q[epoch+'_chat_cases'],gold):
                    assert a['generated_ids']==b['generated_ids']
            phases=q['multi_batch_cases'];assert len(phases)==40
            # Compare all ranks' output IDs and EOS decisions, not unsynchronized clocks.
            signatures=[(c['phase'],c['generated_ids'],c['stopped_at_EOS'])for c in phases]
            if head is None:head=signatures
            assert signatures==head
            for c in phases:
                if c['qa']:
                    key=re.sub(r'-b(?:8|16)(?=-p)', '', c['phase']);gold=refs[key]['generated_ids'];assert all(v==gold[i%8]for i,v in enumerate(c['generated_ids'])),('frozen_QA_complete_IDs',c['phase'])
                elif c['micro'] in[8,16]:
                    gold=refs['g32m8_A1']['generated_ids'];assert all(v==gold[i%8]for i,v in enumerate(c['generated_ids'])),('frozen_natural_complete_IDs',c['phase'])
                else:
                    base=next(v for v in phases if v['phase']=='g1m1_A1');assert c['generated_ids']==base['generated_ids'],('B1_AR_same_configuration',c['phase'])
                batch=c['batch'];micro=c['micro'];gen=c['generated_ids'];times=c['step_completion_seconds'];limit=c['max_output_tokens']
                assert c['all_serving_cards']==CARDS and len(gen)==len(times)==batch
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
                if stage==PP-1:
                    for i,g in enumerate(gen):
                        # Original cohort position and case family define the reference.
                        family=c['phase'].split('-')[1]if c['qa']else i%2
                        key=(stage,rank,c['qa'],family,micro,i%micro,limit)
                        v=g;old=all_refs.setdefault(key,v);assert old==v,('sameM',key,c['phase'])
                if c['qa']:
                    assert all(c['stopped_at_EOS'])
                    if stage==PP-1 and rank==0:
                        assert len(c['answers'])==batch
                        for a in c['answers']:
                            assert a['thinking_closed']and a['known_answer']
                            text=a['answer'];label=c['phase']
                            known=('北京'in text and('中国'in text or'首都'in text))if'beijing'in label else(bool(re.search(r'(?<!\d)3973(?!\d)',text.replace(',','')))if'arithmetic'in label else'待发货'in text)
                            assert known
                elif stage==PP-1 and rank==0:
                    summaries.append({k:c[k]for k in ['phase','batch','micro','pipeline_round_feedback','feedback_calls','elapsed_seconds','aggregate_useful_decode_tokens_s','decode_tokens_s_per_card','including_prefill_useful_tokens_s','including_prefill_tokens_s_per_card','useful_output_tokens','useful_tokens_per_request','TTFT_seconds_per_request','batch_cache_policy','batch_cache_delta','fastpath_bundle_policy','fastpath_counter_delta']})
                    summaries[-1]['input_tokens_per_second_to_all_prefill_complete']=sum(c['input_lengths'])/first
            if stage==PP-1 and rank==0:
                for c in q['raw_checkpoints']:
                    p=r/'results'/c['file'];assert sha(p)==c['sha256'];raw=p.read_bytes();x=(np.frombuffer(raw,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(c['shape'])
                    assert np.isfinite(x).all()and np.argmax(x,axis=1).tolist()==c['GPU_argmax']
                    phase=next(v for v in phases if v['phase']==c['phase'])
                    for j,i in enumerate(c['rows']):assert hashlib.sha256(raw[j*x.shape[1]*2:(j+1)*x.shape[1]*2]).hexdigest()==phase['raw_hashes_per_request'][i][str(c['step'])]
                    cpu_raw+=1
            reports.append({'stage':stage,'rank':rank,'file':f'glm-fastpath-r3772-{step}-stage{stage}-rank{rank}.json','report_sha256':sha(r/'results'/f'glm-fastpath-r3772-{step}-stage{stage}-rank{rank}.json')})
    abba=[]
    for batch,micro in [(1,1),(8,8),(16,8),(16,16)]:
        group=[v for v in summaries if v['batch']==batch and v['micro']==micro]
        assert len(group)==4 and[v['fastpath_bundle_policy']for v in group]==[0,1,1,0]
        import statistics
        A=statistics.median(v['aggregate_useful_decode_tokens_s']for v in group if v['fastpath_bundle_policy']==0)
        B=statistics.median(v['aggregate_useful_decode_tokens_s']for v in group if v['fastpath_bundle_policy']==1)
        abba.append({'batch':batch,'micro':micro,'baseline_total_decode_tok_s_median':A,'candidate_total_decode_tok_s_median':B,'candidate_decode_tok_s_per_card':B/CARDS,'throughput_speed_ratio':B/A,'including_prefill_baseline_median':statistics.median(v['including_prefill_useful_tokens_s']for v in group if v['fastpath_bundle_policy']==0),'including_prefill_candidate_median':statistics.median(v['including_prefill_useful_tokens_s']for v in group if v['fastpath_bundle_policy']==1),'scope':'Native+harness actualsame-process ABBA, all32cards andrealEOS; no HTTP or hardwarepeak claim.'})
    result['ABBA_summary']=abba
    result.update(completed=True,normal_workers_closed=CARDS,CPU_fullvocab_raw_files=cpu_raw,
                  all32_IDs_EOS_schedule_verified=True,all_tail_sameM_token_repeats_verified=True,external_frozen3330_complete_IDs_exact_for_all_multibatch=True,B1_natural_OFF_ON_full_IDs_exact=True,B1_knownQA_complete_IDs_exact_to_frozen999=True,source_pins=build['pins'],logits_bitwise_equality_required=False,transport=mode,actual_transport_logs=transport_logs,
                  performance=summaries,reports=reports)
except BaseException as e:
    import traceback
    result.update(error=repr(e),traceback=traceback.format_exc())
    out.write_text(json.dumps(result,indent=2));raise
out.write_text(json.dumps(result,indent=2));print('GLM_IDLE_R3773_AUDIT_COMPLETE',PP,cpu_raw,flush=True)
