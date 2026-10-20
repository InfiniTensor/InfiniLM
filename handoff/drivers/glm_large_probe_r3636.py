"""Full-model GLM PP cohort probe. Imported by the qualified latest loader."""
import ctypes, hashlib, json, pathlib, time
import torch
from glm_large_schedule_r3627 import groups, accept, metrics

def run(lib, ctx, stage, rank, payload, control, check, ptr, stream, vocab,
        report, report_path, chat_gate):
    root = pathlib.Path('/refactor')
    eos = set(payload['natural_eos_token_ids']);PP=report['PP'];CARDS=report['all_serving_cards'];ROWS=report['cache_rows'];CAPACITY=report['cache_capacity']
    report.update(scope='Latest full78 GLM TP8PP2/4; real EOS, stable cohorts, '
        'Samebinary multi-request boundedmetadata/readonlyprefix/lease bundle OFF/ON/ON/OFF ABBA; batchcachefixedON, roundfeedbackfixed. Public multi-request '
        'paths; same-micro exact IDs/raw repeats; no15 admission gate. '
        '32servicecards;G64/G128/G192 withM4/M8/M16,ctx128or192rows,512input/max128output andtwo frozenprompts repeated. Keep128output separatefrom512output matrix. Native+harness only; no HTTP/diversearrivals/peak claim.',
        multi_batch_cases=[], raw_checkpoints=[], single_user15_is_adoption_gate=False)
    report['probe_source_sha256'] = {
        name: hashlib.sha256((root/'incoming'/name).read_bytes()).hexdigest()
        for name in ['glm_large_schedule_r3627.py', 'glm_large_probe_r3636.py',
                     'modern_glm_large_r3636.py', 'glm_large_r3636_inside.sh']}
    def save():
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(report_path)
    def public(enabled):
        check(lib.modern_glm_model_reset(ctx))
        check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx, 0))
        check(lib.modern_glm_model_feature_bypass_r1406(ctx, 0))
        check(lib.modern_glm_model_decode_leases_policy_r1356(ctx, 0))
        check(lib.modern_glm_model_indexer_policy_r998(ctx, 0))
        if enabled:
            check(lib.modern_glm_model_decode_leases_policy_r1356(ctx,1))
            check(lib.modern_glm_model_indexer_policy_r998(ctx,1))
            check(lib.modern_glm_model_feature_bypass_r1406(ctx,1))
            check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1))
    def batch_stats():
        a=(ctypes.c_uint64*5)();check(lib.modern_glm_model_batch_cache_stats_r3068(ctx,a,5));return list(a)
    cfg=json.loads((pathlib.Path('/old/models/ZhipuAI-GLM-5.3')/'config.json').read_text());bounds=[0,38,78]if PP==2 else[0,18,38,58,78];stage_layers=bounds[stage+1]-bounds[stage];full_layers=sum(x=='full'for x in cfg['indexer_types'][bounds[stage]:bounds[stage+1]])
    def fast_stats():return [int(lib.modern_glm_indexer_metadata_calls_r1406(rank)),int(lib.modern_glm_prefix_calls_r1669(rank,0)),int(lib.modern_glm_prefix_calls_r1669(rank,1))]
    public(0);before=fast_stats();assert lib.modern_glm_model_feature_bypass_r1406(ctx,2)!=0 and lib.modern_glm_model_owned_prefix_policy_r1669(ctx,2)!=0 and fast_stats()==before
    assert lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1)!=0 and fast_stats()==before
    report.update(fastpath_invalid_and_dependency_guards_verified=True,fastpath_stage_layers=stage_layers,fastpath_full_indexer_layers=full_layers)
    references = {}
    def generate(label, prompts, micro, pipelined, limit, answer_eos, qa=False, cache_policy=0):
        public(cache_policy); batch = len(prompts)
        check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1))
        before_policy=batch_stats();assert before_policy[0]==1
        before_fast=fast_stats();report.setdefault("fastpath_phase_flags",{})[label]=cache_policy
        assert batch <= ROWS and all(0 < len(p) and len(p)+limit-1 <= 1024 for p in prompts)
        reserve = torch.cuda.mem_get_info(rank)[0]
        assert reserve > 6*1024**3, ('before_phase_reserve', reserve)
        generated = [[] for _ in prompts]; done = [False]*batch
        completions = [[] for _ in prompts]; hashes = [{} for _ in prompts]
        events = []; feedback_calls = 0; barrier = torch.ones(1, device='cuda')
        control(barrier); assert barrier.item() == CARDS
        start = time.perf_counter(); step = 0
        while not all(done):
            active = [i for i in range(batch) if not done[i]]
            batches = groups(active, micro, step == 0, [len(p)for p in prompts])
            emitted = torch.zeros(batch, device='cuda', dtype=torch.float32)
            saved_ids = {}
            for group in batches:
                descriptions=[]; flat=[]; positions=[]
                for row in group:
                    assert len(generated[row]) == step
                    p = 0 if step == 0 else len(prompts[row])+step-1
                    current = prompts[row] if step == 0 else [generated[row][-1]]
                    descriptions.extend([row, p, len(current)])
                    flat.extend(current if stage == 0 else [0]*len(current))
                    positions.extend(range(p, p+len(current)))
                assert len(flat) <= 4096 and max(positions) < 1024
                ids = torch.tensor(flat, device='cuda', dtype=torch.int64)
                pos = torch.tensor(positions, device='cuda', dtype=torch.int64)
                desc = (ctypes.c_size_t*len(descriptions))(*descriptions)
                logits = torch.full((len(group), vocab), float('nan'),
                                    device='cuda', dtype=torch.bfloat16)
                begin = time.perf_counter()
                check(lib.modern_glm_model_forward(ctx, len(flat), len(group),
                    desc, ptr(ids), ptr(pos), ptr(logits), 0, stream()))
                native = time.perf_counter()-begin
                if not report.get('fastpath_live_sequence_mutation_rejected'):
                    unchanged=fast_stats();assert lib.modern_glm_model_feature_bypass_r1406(ctx,cache_policy)!=0 and lib.modern_glm_model_owned_prefix_policy_r1669(ctx,cache_policy)!=0 and fast_stats()==unchanged
                    report['fastpath_live_sequence_mutation_rejected']=True
                if stage == PP-1:
                    assert torch.isfinite(logits).all(), ('nonfinite', label, step, group)
                    chosen = logits.argmax(-1)
                    chosen_cpu = [int(v) for v in chosen.cpu().tolist()]
                    if step in [0,1,2,3,63,127,255,383,511]:
                        raw = logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        if rank == 0:
                            name = f'glm-fastpath-r3627-{report["step"]}-{label}-s{step}-r{group[0]}-bf16.bin'
                            path = root/'results'/name; assert not path.exists(); path.write_bytes(raw)
                            report['raw_checkpoints'].append({'file': name, 'rows': group,
                                'shape': [len(group), vocab], 'phase': label, 'step': step,
                                'sha256': hashlib.sha256(raw).hexdigest(), 'GPU_argmax': chosen_cpu})
                        for offset, row in enumerate(group):
                            b = raw[offset*vocab*2:(offset+1)*vocab*2]
                            hashes[row][str(step)] = hashlib.sha256(b).hexdigest()
                    for row, token in zip(group, chosen_cpu): saved_ids[row] = token
                    if rank == 0: emitted[group] = chosen.float()
                if not pipelined:
                    packet = emitted[group].contiguous(); control(packet); feedback_calls += 1
                    values = [int(v) for v in packet.cpu().tolist()]
                    assert all(0 <= v < vocab for v in values)
                    if stage == PP-1: assert values == [saved_ids[i] for i in group]
                    stamp = time.perf_counter()-start
                    accept(generated, done, group, values, answer_eos, limit)
                    for row in group: completions[row].append(stamp)
                events.append({'round': step, 'rows': group, 'native_seconds': native})
            if pipelined:
                control(emitted); feedback_calls += 1
                values = [int(v) for v in emitted[active].cpu().tolist()]
                assert all(0 <= v < vocab for v in values)
                if stage == PP-1: assert values == [saved_ids[i] for i in active]
                stamp = time.perf_counter()-start
                accept(generated, done, active, values, answer_eos, limit)
                for row in active: completions[row].append(stamp)
            if step % 32 == 0:
                report['phase_progress'] = {'label': label, 'round': step,
                    'active_requests': len(active), 'elapsed_seconds': time.perf_counter()-start}
                save(); print('GLM_COHORT_R3024_PROGRESS', stage, rank,
                              report['phase_progress'], flush=True)
            step += 1
        elapsed = time.perf_counter()-start
        # One EOS or limit terminal token per request; no forward occurs after it.
        assert all(g[-1] in answer_eos or len(g) == limit for g in generated)
        row = {'phase': label, 'batch': batch, 'micro': micro,
            'pipeline_round_feedback': pipelined, 'max_output_tokens': limit,
            'input_lengths': [len(p) for p in prompts], 'generated_ids': generated,
            'raw_hashes_per_request': hashes, 'step_completion_seconds': completions,
            'events': events, 'feedback_calls': feedback_calls, 'elapsed_seconds': elapsed,
            'stopped_at_EOS': [g[-1] in answer_eos for g in generated],
            'no_forward_after_EOS': True, 'qa': qa,
            'memory_free_after_phase': torch.cuda.mem_get_info(rank)[0],
            **metrics(generated, completions, answer_eos, elapsed, CARDS)}
        report['last_completed_generation']=row;save()
        if qa:
            assert all(row['stopped_at_EOS']), ('QA missing EOS', label)
            if stage == PP-1 and rank == 0:
                from glm_chat_template_r20 import prepare_tokenizer, decode_answer
                import re
                tokenizer, kwargs, provenance = prepare_tokenizer()
                assert provenance == payload['chat_template_provenance']
                case = next(c for c in payload['chat_cases'] if label.startswith('QA-'+c['label']+'-'))
                row['answers'] = []
                for g in generated:
                    text, raw_text, closed = decode_answer(tokenizer, g, case['generation_starts_in_thinking'])
                    known = ('北京' in text and ('中国' in text or '首都' in text)) if case['label']=='beijing' else (bool(re.search(r'(?<!\d)3973(?!\d)', text.replace(',',''))) if case['label']=='arithmetic' else '待发货' in text)
                    assert known and closed, ('multi_batch_QA', label, text)
                    row['answers'].append({'answer': text, 'thinking_closed': closed, 'known_answer': known})
        # Identical initial micro cohort and per-position prompts; EOS filtering
        # preserves cohort shape. Compare every sequence and checkpoint byte hash.
        for i, (p, g) in enumerate(zip(prompts, generated)):
            key = (qa, micro, i % micro, hashlib.sha256(json.dumps(p).encode()).hexdigest(), limit)
            if key in references:
                gold = references[key]
                assert g == gold['ids'], ('same_micro_IDs', label, i)
                report.setdefault('raw_repeat_diagnostics',[]).append({'phase':label,'row':i,'same_micro_raw':hashes[i]==gold['raw']})if stage==PP-1 else None
            else: references[key] = {'ids': g, 'raw': hashes[i]}
        after_policy=batch_stats();delta=[b-a for a,b in zip(before_policy,after_policy)]
        assert after_policy[0]==1 and delta[4]==0
        after_fast=fast_stats();fast_delta=[b-a for a,b in zip(before_fast,after_fast)]
        decode_calls=sum(e['round']>0 for e in events);decode_requests=sum(len(e['rows'])for e in events if e['round']>0)
        expected=[full_layers*decode_calls,full_layers*decode_calls,stage_layers*decode_requests]if cache_policy else[0,0,0]
        assert fast_delta==expected,('actualfastpathdispatch',label,fast_delta,expected)
        row.update(fastpath_bundle_policy=cache_policy,fastpath_counters_before=before_fast,fastpath_counters_after=after_fast,fastpath_counter_delta=fast_delta,batch_cache_policy=1,batch_cache_before=before_policy,
                   batch_cache_after=after_policy,batch_cache_delta=delta)
        row['same_micro_reference_passed'] = True
        report['multi_batch_cases'].append(row); save()
        print('GLM_COHORT_R3024_PHASE', stage, rank, label,
              row['aggregate_useful_decode_tokens_s'], flush=True)

    # Validate every tested microgeometry before its long phases.
    for qa_batch in[4,8,16]:
        for case in payload['chat_cases']:
            for repeat in[1,2]:
                generate('QA-'+case['label']+'-b'+str(qa_batch)+'-p1-r'+str(repeat),[case['input_ids']]*qa_batch,qa_batch,True,256,set(payload['chat_eos_token_ids']),True,1)
    natural=[c['input_ids']for c in payload['natural_cases']];assert len(natural)==2
    configs=[(b,m)for m in[4,8,16]for b in[64,128,192]if b<=ROWS]
    for repetition,order in[(1,configs),(2,list(reversed(configs)))]:
        for batch,micro in order:
            generate('g'+str(batch)+'m'+str(micro)+'_r'+str(repetition),[natural[i%2]for i in range(batch)],micro,True,128,eos,cache_policy=1)
    # Restore qualified B1 policies and frozen quality checks after batch work.
    check(lib.modern_glm_model_reset(ctx))
    check(lib.modern_glm_model_indexer_policy_r998(ctx,1))
    check(lib.modern_glm_model_decode_leases_policy_r1356(ctx,1))
    check(lib.modern_glm_model_feature_bypass_r1406(ctx,1))
    check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1))
    chat_gate('after_select')
    report['cohort_phases_completed'] = len(report['multi_batch_cases']) == 18+2*len(configs)
    assert report['cohort_phases_completed']; save()
