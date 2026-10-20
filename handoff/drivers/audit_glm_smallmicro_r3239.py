import datetime, hashlib, json, math, pathlib, re, subprocess
import numpy as np
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/glm-g128-r3239-audit.json';assert not out.exists()
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,
        'scope':'Independent CPU audit of terminal GLM32card G64/G128/M8 PP4eight/sixteencohort experiment;ctx128,512input/max128output andtwofrozenprompts repeated, separatefrom512output results; '
        'all32 reports/fullIDs/EOS/CPUfullvocabraw/checksums/sameMrepeat/normalclose. '
        'Native+harness timing, no HTTP or hardware-peak claim.'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
try:
    rows=[a.split('|')for a in subprocess.check_output(['sacct','-j','910379','-n','-P','-o','JobID,JobName%110,State,ExitCode,NodeList%160'],text=True).splitlines()]
    rows=[a for a in rows if a[1]=='models-latest-r3237-GLMG64G128M8ABBA-910379']
    assert len(rows)==1;state=rows[0];result['producer_state']=state
    assert state[2:4]==['COMPLETED','0:0'],state
    step=state[0];reports=[];cpu_raw=0;head=None
    natural=load(r/'results'/f'glm-serving-cases-r2130-{step}.json');eos=set(natural['eos_token_ids'])
    sdk=load(r/'results/latest-sdk-r2844-GLM-build.json');build=load(r/'results/glm-batch-model-r3073-build.json')
    common=load(r/'results/common-latest-r2845-GLM-build.json')
    all_refs={};summaries=[]
    for stage in range(4):
        for rank in range(8):
            p=r/'results'/f'glm-smallmicro-r3236-{step}-stage{stage}-rank{rank}.json';q=load(p)
            assert q['completed']and q['worker_shutdown_completed']and q['cohort_phases_completed']and q['cache_rows']==128
            assert q['library_sha256']==build['library_sha256']and q['actual_sdk_sha256']==sdk['library_sha256']
            assert q['LM_source_commit']=='3e0d89fe881ca9fb2411c6e35ffc913c7ea6654d'
            assert q['batch_cache_default_OFF_verified']and q['batch_cache_wrong_thread_and_invalid_untouched']and q['batch_cache_live_sequence_policy_rejected']
            assert len(q['actual_FP8_helpers'])==2
            for name,h in q['actual_FP8_helpers'].items():
                assert '/candidates/glm-batch-model-r3072/helpers/'in name
                assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h==build['helpers']['helpers/'+pathlib.Path(name).name]

            for name,h in q['probe_source_sha256'].items():assert sha(r/'incoming'/name)==h
            for name,h in q['actual_LM_native_libraries'].items():
                assert sha(pathlib.Path(name.replace('/refactor',str(r),1)))==h
                if 'libinfinilm_runtime.so'in name:assert h==common['runtime_sha256']
            assert q['probe_preimport_before_CCL_verified']and q['Python_bytecode_cache_prefix'].startswith('/local/infini-pycache-')
            assert q['owner_affinity']['verified']and q['owner_affinity']['width']==4,q['owner_affinity']
            assert q['BLAS_policy']['failures']==0 and all(v['created']==v['verified']for v in q['BLAS_policy']['owners'])
            after=q['cache_after_destroy'];assert after['created']==after['destroyed']and after['active']==after['active_bytes']==0
            for epoch in ['before_select','after_select']:
                assert q[epoch+'_chat_gate_passed']
                gold=load(r/'results'/f'glm-fp8-cached-model-r999-910374.89-stage{stage}-rank{rank}.json')[epoch+'_chat_cases']
                for a,b in zip(q[epoch+'_chat_cases'],gold):
                    assert a['generated_ids']==b['generated_ids']and a['first4_logits_sha256']==b['first4_logits_sha256']
            phases=q['multi_batch_cases'];assert len(phases)==17
            # Compare all ranks' output IDs and EOS decisions, not unsynchronized clocks.
            signatures=[(c['phase'],c['generated_ids'],c['stopped_at_EOS'])for c in phases]
            if head is None:head=signatures
            assert signatures==head
            for c in phases:
                batch=c['batch'];micro=c['micro'];gen=c['generated_ids'];times=c['step_completion_seconds'];limit=c['max_output_tokens']
                assert c['all_serving_cards']==32 and len(gen)==len(times)==batch
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
                if c['batch_cache_policy']==0:assert delta[1:4]==[0,0,0]
                elif micro>=32 and any(e['round']>0 and len(e['rows'])==micro for e in c['events']):assert delta[1 if micro==32 else 2]>0

                assert c['feedback_calls']==(len(per_round)if c['pipeline_round_feedback']else feedback)
                useful=sum(len(g)-int(g[-1]in answer_eos)for g in gen)
                first=max(ts[0]for ts in times);finish=max(ts[-1]for ts in times)
                dec=sum(sum(t>first and tok not in answer_eos for t,tok in zip(ts,g))for ts,g in zip(times,gen))
                assert useful==c['useful_output_tokens']and dec==c['decode_interval_useful_tokens']
                rate=dec/(finish-first)if finish>first else None
                if rate is not None:assert math.isclose(rate,c['aggregate_useful_decode_tokens_s'],rel_tol=1e-12)
                assert math.isclose(useful/c['elapsed_seconds'],c['including_prefill_useful_tokens_s'],rel_tol=1e-12)
                if stage==3:
                    for i,g in enumerate(gen):
                        # Original cohort position and case family define the reference.
                        family=c['phase'].split('-')[1]if c['qa']else i%2
                        key=(stage,rank,c['qa'],family,micro,i%micro,limit)
                        v=(g,c['raw_hashes_per_request'][i]);old=all_refs.setdefault(key,v);assert old==v,('sameM',key,c['phase'])
                if c['qa']:
                    assert all(c['stopped_at_EOS'])
                    if stage==3 and rank==0:
                        assert len(c['answers'])==batch
                        for a in c['answers']:
                            assert a['thinking_closed']and a['known_answer']
                            text=a['answer'];label=c['phase']
                            known=('北京'in text and('中国'in text or'首都'in text))if'beijing'in label else(bool(re.search(r'(?<!\d)3973(?!\d)',text.replace(',','')))if'arithmetic'in label else'待发货'in text)
                            assert known
                elif stage==3 and rank==0:
                    summaries.append({k:c[k]for k in ['phase','batch','micro','pipeline_round_feedback','feedback_calls','elapsed_seconds','aggregate_useful_decode_tokens_s','decode_tokens_s_per_card','including_prefill_useful_tokens_s','including_prefill_tokens_s_per_card','useful_output_tokens','useful_tokens_per_request','TTFT_seconds_per_request','batch_cache_policy','batch_cache_delta']})
                    summaries[-1]['input_tokens_per_second_to_all_prefill_complete']=sum(c['input_lengths'])/first
            if stage==3 and rank==0:
                for c in q['raw_checkpoints']:
                    p=r/'results'/c['file'];assert sha(p)==c['sha256'];raw=p.read_bytes();x=(np.frombuffer(raw,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(c['shape'])
                    assert np.isfinite(x).all()and np.argmax(x,axis=1).tolist()==c['GPU_argmax']
                    phase=next(v for v in phases if v['phase']==c['phase'])
                    for j,i in enumerate(c['rows']):assert hashlib.sha256(raw[j*x.shape[1]*2:(j+1)*x.shape[1]*2]).hexdigest()==phase['raw_hashes_per_request'][i][str(c['step'])]
                    cpu_raw+=1
            reports.append({'stage':stage,'rank':rank,'file':f'glm-smallmicro-r3236-{step}-stage{stage}-rank{rank}.json','report_sha256':sha(r/'results'/f'glm-smallmicro-r3236-{step}-stage{stage}-rank{rank}.json')})
    import statistics
    system=[v for v in summaries if v['phase'] in ['g64m8_A1','g128m8_B1','g128m8_B2','g64m8_A2']]
    assert len(system)==4 and [v['batch']for v in system]==[64,128,128,64]
    assert all(v['micro']==8 and v['batch_cache_policy']==0 and v['pipeline_round_feedback']for v in system)
    A=statistics.median(v['aggregate_useful_decode_tokens_s']for v in system if v['batch']==64)
    B=statistics.median(v['aggregate_useful_decode_tokens_s']for v in system if v['batch']==128)
    result['ABBA_summary']={'G64_total_decode_median':A,'G128_total_decode_median':B,'ratio':B/A,'G128_decode_per_all32cards':B/32,'phases':system,'scope':'512input/max128output. FixedM8/cache/GPUmath androundfeedback; two frozenprompts repeated. Not comparableto512output noronline diversearrival result.'}
    result.update(completed=True,normal_workers_closed=32,CPU_fullvocab_raw_files=cpu_raw,
                  all32_IDs_EOS_schedule_verified=True,all_tail_sameM_raw_repeats_verified=True,
                  performance=summaries,reports=reports)
except BaseException as e:
    import traceback
    result.update(error=repr(e),traceback=traceback.format_exc())
    out.write_text(json.dumps(result,indent=2));raise
out.write_text(json.dumps(result,indent=2));print('GLM_COHORT_R3034_AUDIT_COMPLETE',cpu_raw,flush=True)
