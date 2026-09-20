from collections import Counter
from pathlib import Path

from scripts.benchmark_scheduler_admission import (
    build_scheduler,
    exact_extra_blocks,
    legacy_extra_blocks,
    scan_reservation,
)
from scripts.benchmark_serving_admission import predict_pressure
from scripts.kv_admission_manifest import load_manifest

FIXED_MANIFEST = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "kv_admission_tinyllama_fixed.json"
)
FIXED_MANIFEST_SHA256 = (
    "b12ca11d8e36dfc7fb88d8946234607e888f7161cbb30907d38ca4ca1db72bca"
)


def test_fixed_workload_drives_all_three_validation_layers():
    manifest = load_manifest(FIXED_MANIFEST)

    assert manifest.sha256 == FIXED_MANIFEST_SHA256
    assert manifest.model == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert manifest.block_size == 256
    assert manifest.num_blocks == 864
    assert len(manifest.wave1) == 256
    assert len(manifest.wave2) == 128
    assert Counter(item.prompt_tokens for item in manifest.requests) == {
        64: 48,
        128: 48,
        192: 48,
        256: 48,
        320: 48,
        512: 48,
        768: 48,
        1024: 48,
    }
    assert {item.max_tokens for item in manifest.requests} == {128}

    scheduler, running = build_scheduler(manifest)
    legacy_reserved = scan_reservation(running, 256, legacy_extra_blocks)
    exact_reserved = scan_reservation(running, 256, exact_extra_blocks)
    pressure = predict_pressure(manifest)

    assert legacy_reserved == 256
    assert exact_reserved == 160
    assert pressure["false_reserved_blocks"] == 96
    assert pressure["predicted_wave2_admissions_legacy"] == 53
    assert pressure["predicted_wave2_admissions_exact"] == 91
    tracked = scheduler.get_cache_stats().get("num_reserved_decode_blocks")
    if tracked is not None:
        assert tracked == exact_reserved
