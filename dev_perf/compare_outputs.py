#!/usr/bin/env python3
"""Compare output_token_ids across bench JSON files (greedy decoding).

Usage: compare_outputs.py file_a.json file_b.json [file_c.json ...]

For each workload present in all files, reports per-request token match:
exact match count, and for mismatches the first divergence position.
Slimmed files carry output_token_ids_sha256 instead of full ids; those are
compared by hash (exact match only, no divergence position). A full-ids
file compared against a hash-only one is hashed on the fly with the same
digest (sha256 of the compact JSON encoding), so archived files stay
comparable with fresh --dump-outputs runs.
"""
import hashlib
import json
import sys


def digest(ids):
    payload = json.dumps(ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load(path):
    d = json.load(open(path))
    label = f"{d['engine']}/{d.get('attn_backend', '-')}"
    wl = {}
    for w in d["workloads"]:
        ids = w.get("output_token_ids")
        wl[w["workload"]] = ids if ids is not None else w.get("output_token_ids_sha256")
    return label, wl


def main(paths):
    loaded = [load(p) for p in paths]
    base_label, base_wl = loaded[0]
    print(f"reference: {base_label} ({paths[0]})")
    for label, wl in loaded[1:]:
        print(f"\n=== {label} vs {base_label} ===")
        for name, ref_ids in base_wl.items():
            ids = wl.get(name)
            if ref_ids is None or ids is None:
                print(f"  {name}: missing token ids, skipped")
                continue
            if not ref_ids or not ids:
                print(f"  {name}: empty request list, skipped")
                continue
            same_count = len(ref_ids) == len(ids)
            if not same_count:
                print(
                    f"  {name}: request count differs "
                    f"({len(ref_ids)} vs {len(ids)}), "
                    f"comparing first {min(len(ref_ids), len(ids))}"
                )
            if isinstance(ref_ids[0], str) != isinstance(ids[0], str):
                # mixed full-vs-hash: hash the full side
                if isinstance(ids[0], str):
                    ref_ids = [digest(r) for r in ref_ids]
                else:
                    ids = [digest(r) for r in ids]
            hashed = isinstance(ref_ids[0], str)
            suffix = " (sha256)" if hashed else ""
            n_exact = 0
            worst_div = None
            for a, b in zip(ref_ids, ids):
                if a == b:
                    n_exact += 1
                    continue
                if hashed:
                    continue
                div = next(
                    (j for j, (x, y) in enumerate(zip(a, b)) if x != y),
                    min(len(a), len(b)),
                )
                if worst_div is None or div < worst_div:
                    worst_div = div
            total = len(ref_ids)
            if n_exact == total and same_count:
                print(f"  {name}: {total}/{total} requests exact match{suffix}")
            elif hashed:
                print(f"  {name}: {n_exact}/{total} exact{suffix}")
            elif worst_div is not None:
                print(
                    f"  {name}: {n_exact}/{total} exact, "
                    f"earliest divergence at token {worst_div}"
                )
            else:
                # mismatch purely from the request-count difference
                print(f"  {name}: {n_exact}/{total} exact (compared prefixes identical)")
        for name in wl:
            if name not in base_wl:
                print(f"  {name}: only in {label}, not compared")


if __name__ == "__main__":
    main(sys.argv[1:])
