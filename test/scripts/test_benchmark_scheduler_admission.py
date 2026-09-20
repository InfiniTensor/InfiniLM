import json

from scripts.benchmark_scheduler_admission import (
    benchmark,
    exact_extra_blocks,
    legacy_extra_blocks,
    make_request,
)
from scripts.kv_admission_manifest import load_manifest


def test_fixed_manifest_drives_scheduler_formula(tmp_path):
    path = tmp_path / "manifest.json"
    requests = []
    for wave, count in (("wave1", 4), ("wave2", 2)):
        for index in range(count):
            length = 17 + index * 5
            requests.append(
                {
                    "request_id": f"{wave}-{index:04d}",
                    "wave": wave,
                    "prompt": "x " * length,
                    "prompt_token_ids": list(range(length)),
                    "max_tokens": 15,
                }
            )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configuration": {
                    "model": "test-model",
                    "block_size": 16,
                    "num_blocks": 32,
                    "seed": 1,
                },
                "requests": requests,
            }
        ),
        encoding="utf-8",
    )
    manifest = load_manifest(path)

    request = make_request(manifest.wave1[0])
    request.block_table = [0, 1]
    assert legacy_extra_blocks(request, 16) == 1
    assert exact_extra_blocks(request, 16) == 0

    result = benchmark(manifest, "incremental", iterations=3, repeats=1)
    assert result.manifest_sha256 == manifest.sha256
    assert result.running_requests == 4
    assert result.candidate_requests == 2
    assert 0 <= result.accepted_candidates <= 2
    assert result.scheduler_reserved_blocks == result.exact_reserved_blocks
    assert result.admission_median_us > 0
