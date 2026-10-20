"""Full-model GLM PP cohort probe. Imported by the qualified latest loader."""
import ctypes, hashlib, json, pathlib, time
import torch
from glm_cohort_schedule_r3200 import groups, accept, metrics

def run(lib, ctx, stage, rank, payload, control, check, ptr, stream, vocab,
        report, report_path, chat_gate):
    root = pathlib.Path('/refactor')
    eos = set(payload['natural_eos_token_ids'])
    report.update(scope='Latest full78 GLM TP8PP4; real EOS, stable cohorts, '
        'Private3348 prefillstableexpertorder OFF/ON/ON/OFF ABBA; batchcachefixedON, roundfeedbackfixed. Public multi-request '
        'paths; same-micro exact IDs/raw repeats; no15 admission gate. '
        '32servicecards;G64/G128M8 withctx64rows,512input/max128output andtwo frozenprompts repeated. Keep128output separatefrom512output matrix. Native+harness only; no HTTP/diversearrivals/peak claim.',
        multi_batch_cases=[], raw_checkpoints=[], single_user15_is_adoption_gate=False)
    report['probe_source_sha256'] = {
        name: hashlib.sha256((root/'incoming'/name).read_bytes()).hexdigest()
        for name in ['glm_cohort_schedule_r3200.py', 'glm_expert_order_probe_r3366.py',
                     'modern_glm_expert_order_r3366.py', 'glm_expert_order_r3366_inside.sh']}
    def save():
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(report_path)
    def public(enabled):
        check(lib.modern_glm_model_reset(ctx))
        check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx, 0))
        check(lib.modern_glm_model_feature_bypass_r1406(ctx, 0))
        check(lib.modern_glm_model_decode_leases_policy_r1356(ctx, 0))
        check(lib.modern_glm_model_indexer_policy_r998(ctx, 0))
        check(lib.modern_glm_model_expert_order_policy_r3348(ctx,enabled))
    def batch_stats():
        a=(ctypes.c_uint64*5)();check(lib.modern_glm_model_batch_cache_stats_r3068(ctx,a,5));return list(a)
    bounds=[0,18,38,58,78];sparse_layers=bounds[stage+1]-max(3,bounds[stage])
    def order_stats():
        a=(ctypes.c_uint64*6)();check(lib.modern_glm_model_expert_order_stats_r3348(ctx,a,6));return list(a)
    public(0);before=order_stats();assert lib.modern_glm_model_expert_order_policy_r3348(ctx,2)!=0 and order_stats()==before
    report.update(expert_order_invalid_guard_verified=True,expert_order_sparse_layers=sparse_layers)
    references = {}
    def generate(label, prompts, micro, pipelined, limit, answer_eos, qa=False, cache_policy=0):
        public(cache_policy); batch = len(prompts)
        check(lib.modern_glm_model_batch_cache_policy_r3068(ctx,1))
        before_policy=batch_stats();assert before_policy[0]==1
        before_order=order_stats();report.setdefault("expert_order_phase_flags",{})[label]=cache_policy
        assert batch <= 128 and all(0 < len(p) and len(p)+limit-1 <= 1024 for p in prompts)
        reserve = torch.cuda.mem_get_info(rank)[0]
        assert reserve > 6*1024**3, ('before_phase_reserve', reserve)
        generated = [[] for _ in prompts]; done = [False]*batch
        completions = [[] for _ in prompts]; hashes = [{} for _ in prompts]
        events = []; feedback_calls = 0; barrier = torch.ones(1, device='cuda')
        control(barrier); assert barrier.item() == 32
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
                if not report.get('expert_order_live_sequence_mutation_rejected'):
                    unchanged=order_stats();assert lib.modern_glm_model_expert_order_policy_r3348(ctx,cache_policy)!=0 and order_stats()==unchanged
                    report['expert_order_live_sequence_mutation_rejected']=True
                if stage == 3:
                    assert torch.isfinite(logits).all(), ('nonfinite', label, step, group)
                    chosen = logits.argmax(-1)
                    chosen_cpu = [int(v) for v in chosen.cpu().tolist()]
                    if step in [0,1,2,3,63,127,255,383,511]:
                        raw = logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        if rank == 0:
                            name = f'glm-expert-order-r3366-{report["step"]}-{label}-s{step}-r{group[0]}-bf16.bin'
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
                    if stage == 3: assert values == [saved_ids[i] for i in group]
                    stamp = time.perf_counter()-start
                    accept(generated, done, group, values, answer_eos, limit)
                    for row in group: completions[row].append(stamp)
                events.append({'round': step, 'rows': group, 'native_seconds': native})
            if pipelined:
                control(emitted); feedback_calls += 1
                values = [int(v) for v in emitted[active].cpu().tolist()]
                assert all(0 <= v < vocab for v in values)
                if stage == 3: assert values == [saved_ids[i] for i in active]
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
            **metrics(generated, completions, answer_eos, elapsed, 32)}
        if qa:
            assert all(row['stopped_at_EOS']), ('QA missing EOS', label)
            if stage == 3 and rank == 0:
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
                if stage == 3: assert hashes[i] == gold['raw'], ('same_micro_raw', label, i)
            else: references[key] = {'ids': g, 'raw': hashes[i]}
        after_policy=batch_stats();delta=[b-a for a,b in zip(before_policy,after_policy)]
        assert after_policy[0]==1 and delta[4]==0
        after_order=order_stats();order_delta=[b-a for a,b in zip(before_order,after_order)]
        eligible=[sum(len(prompts[i])for i in e['rows'])if e['round']==0 else len(e['rows'])for e in events]
        eligible=[n for n in eligible if n>=512];calls=sparse_layers*len(eligible)if cache_policy else 0
        expected=[0,calls,calls,calls,sparse_layers*sum(eligible)*8 if cache_policy else 0]
        assert order_delta[:5]==expected and before_order[0]==after_order[0]==cache_policy,('actualprefilldispatch',label,order_delta,expected)
        assert after_order[5]==max([before_order[5]]+([n*64 for n in eligible]if cache_policy else[]))
        row.update(expert_order_policy=cache_policy,expert_order_counters_before=before_order,expert_order_counters_after=after_order,expert_order_counter_delta=order_delta,batch_cache_policy=1,batch_cache_before=before_policy,batch_cache_after=after_policy,batch_cache_delta=delta)
        row['same_micro_reference_passed'] = True
        report['multi_batch_cases'].append(row); save()
        print('GLM_COHORT_R3024_PHASE', stage, rank, label,
              row['aggregate_useful_decode_tokens_s'], flush=True)

    # Full QA32 repeats on both numerical-identical policy paths before long phases.
    for cache_policy in [0,1]:
        for case in payload['chat_cases']:
            for repeat in [1,2]:
                generate('QA-'+case['label']+'-p'+str(cache_policy)+'-r'+str(repeat),
                         [case['input_ids']]*32,8,True,256,
                         set(payload['chat_eos_token_ids']),True,cache_policy)
    natural=[c['input_ids']for c in payload['natural_cases']];assert len(natural)==2
    for batch in[64,128]:
        for suffix,enabled in [('A1',0),('B1',1),('B2',1),('A2',0)]:
            generate('g'+str(batch)+'m8_'+suffix,[natural[i%2]for i in range(batch)],8,True,128,eos,cache_policy=enabled)
    check(lib.modern_glm_model_reset(ctx));check(lib.modern_glm_model_expert_order_policy_r3348(ctx,0))
    # Restore qualified B1 policies and frozen quality checks after batch work.
    check(lib.modern_glm_model_reset(ctx))
    check(lib.modern_glm_model_indexer_policy_r998(ctx,1))
    check(lib.modern_glm_model_decode_leases_policy_r1356(ctx,1))
    check(lib.modern_glm_model_feature_bypass_r1406(ctx,1))
    check(lib.modern_glm_model_owned_prefix_policy_r1669(ctx,1))
    chat_gate('after_select')
    report['cohort_phases_completed'] = len(report['multi_batch_cases']) == 20
    assert report['cohort_phases_completed']; save()
