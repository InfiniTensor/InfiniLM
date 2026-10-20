"""Full-model GLM PP cohort probe. Imported by the qualified latest loader."""
import ctypes, hashlib, json, pathlib, time
import torch
from glm_cohort_schedule_r3048 import groups, accept, metrics

def run(lib, ctx, stage, rank, payload, control, check, ptr, stream, vocab,
        report, report_path, chat_gate):
    root = pathlib.Path('/refactor')
    eos = set(payload['natural_eos_token_ids'])
    report.update(scope='Historical r927 full78 GLM TP8PP4; real EOS, stable cohorts, '
        'serial per-micro vs one-feedback-per-round ABBA. Public multi-request '
        'paths; same-micro exact IDs/raw repeats; no15 admission gate. '
        '32 service cards, host/native harness timing, no HTTP or peak claim.',
        multi_batch_cases=[], raw_checkpoints=[], single_user15_is_adoption_gate=False)
    report['probe_source_sha256'] = {
        name: hashlib.sha256((root/'incoming'/name).read_bytes()).hexdigest()
        for name in ['glm_cohort_schedule_r3048.py', 'legacy_glm_eos_probe_r3429.py',
                     'legacy_glm_eos_r3429.py', 'legacy_glm_eos_r3429_inside.sh']}
    def save():
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)); tmp.replace(report_path)
    def public():
        check(lib.modern_glm_model_reset(ctx))
    references = {}
    def generate(label, prompts, micro, pipelined, limit, answer_eos, qa=False):
        public(); batch = len(prompts)
        assert batch <= 64 and all(0 < len(p) and len(p)+limit-1 <= 1024 for p in prompts)
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
                if stage == 3:
                    assert torch.isfinite(logits).all(), ('nonfinite', label, step, group)
                    chosen = logits.argmax(-1)
                    chosen_cpu = [int(v) for v in chosen.cpu().tolist()]
                    if step in [0,1,2,3,63,127,255,383,511]:
                        raw = logits.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                        if rank == 0:
                            name = f'legacy-glm-eos-r3429-{report["step"]}-{label}-s{step}-r{group[0]}-bf16.bin'
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
        row['same_micro_reference_passed'] = True
        report['multi_batch_cases'].append(row); save()
        print('GLM_COHORT_R3024_PHASE', stage, rank, label,
              row['aggregate_useful_decode_tokens_s'], flush=True)

    # Exercise multi-row quality and repeats before long performance phases.
    for case in payload['chat_cases']:
        for repeat in [1,2]:
            generate('QA-'+case['label']+'-'+str(repeat), [case['input_ids']]*8,
                     8, True, 256, set(payload['chat_eos_token_ids']), True)
    natural = [c['input_ids'] for c in payload['natural_cases']]
    assert len(natural) == 2
    phases = [('g16m16_ref1',16,16,True),('g16m16_ref2',16,16,True),('g64m16_r1',64,16,True),('g64m16_r2',64,16,True),('g32m8_ref1',32,8,True),('g32m8_ref2',32,8,True),('g64m8_r1',64,8,True),('g64m8_r2',64,8,True)]
    for label, batch, micro, pipeline in phases:
        generate(label, [natural[i%2] for i in range(batch)], micro, pipeline, 512, eos)
    # Restore qualified B1 policies and frozen quality checks after batch work.
    check(lib.modern_glm_model_reset(ctx))
    chat_gate('after_performance')
    report['cohort_phases_completed'] = len(report['multi_batch_cases']) == 14
    assert report['cohort_phases_completed']; save()
