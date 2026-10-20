"""Pure host schedule: stable cache rows, EOS retirement before next forward."""
def groups(rows, micro, prefill=False, input_lengths=None):
    assert 1 <= micro <= 64 and rows == sorted(set(rows))
    # Preserve initial cohort membership when some requests finish early.
    out = []
    for base in range(0, 192, micro):
        cohort = [v for v in rows if base <= v < base + micro]
        width = min(micro, 8) if prefill else micro
        if prefill and cohort and input_lengths is not None:
            longest=max(input_lengths[i]for i in cohort)
            assert 0 < longest <= 1024
            width=min(width,4096//longest)
            assert width>=1
        out += [cohort[i:i + width] for i in range(0, len(cohort), width)]
    return out

def accept(generated, done, rows, values, eos, limit):
    assert len(rows) == len(values) and len(rows) == len(set(rows))
    assert all(not done[i] and len(generated[i]) < limit for i in rows)
    for i, value in zip(rows, values):
        generated[i].append(int(value))
        done[i] = value in eos or len(generated[i]) == limit

def metrics(generated, completions, eos, elapsed, cards):
    useful = [len(g) - int(g[-1] in eos) for g in generated]
    first = max(t[0] for t in completions)
    finish = max(t[-1] for t in completions)
    # All rows have finished prefill before this common decode interval starts.
    # Count only useful tokens whose actual feedback completion follows it.
    decode = sum(sum(t > first and token not in eos for t, token in zip(ts, g))
                 for ts, g in zip(completions, generated))
    rate = decode / (finish - first) if finish > first else None
    return {'useful_output_tokens': sum(useful), 'all_prefill_complete_seconds': first,
            'last_completion_seconds': finish, 'decode_interval_useful_tokens': decode,
            'aggregate_useful_decode_tokens_s': rate,
            'decode_tokens_s_per_card': rate / cards if rate is not None else None,
            'including_prefill_useful_tokens_s': sum(useful) / elapsed,
            'including_prefill_tokens_s_per_card': sum(useful) / elapsed / cards,
            'useful_tokens_per_request': useful, 'all_serving_cards': cards,
            'TTFT_seconds_per_request': [ts[0] for ts in completions],
            'request_useful_decode_tokens_s': [
                (n - 1) / (ts[-1] - ts[0]) if n > 1 else None
                for n, ts in zip(useful, completions)]}
