from scripts.benchmark_serving_admission import (
    FirstTokenBarrier,
    metric_summary,
    predict_pressure,
)
from scripts.kv_admission_manifest import AdmissionManifest, RequestSpec


def make_manifest(tmp_path):
    requests = tuple(
        RequestSpec(
            request_id=f"wave{wave}-{index:04d}",
            wave=wave,
            prompt="prompt",
            prompt_token_ids=tuple(range(length)),
            max_tokens=8,
            ordinal=ordinal,
        )
        for ordinal, (wave, index, length) in enumerate(
            (("wave1", 0, 17), ("wave1", 1, 32), ("wave2", 0, 17))
        )
    )
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    return AdmissionManifest(
        path=path,
        model="test-model",
        block_size=16,
        num_blocks=16,
        seed=1,
        requests=requests,
        metadata={},
    )


def test_pressure_uses_the_same_wave_records(tmp_path):
    pressure = predict_pressure(make_manifest(tmp_path))

    assert pressure["wave1_prompt_blocks"] == 4
    assert pressure["legacy_running_reserved_blocks"] == 2
    assert pressure["exact_running_reserved_blocks"] == 1


def test_barrier_becomes_ready_at_target():
    barrier = FirstTokenBarrier(target=2, total=3)
    barrier.mark_first_token()
    assert not barrier.event.is_set()
    barrier.mark_first_token()
    assert barrier.reached
    assert barrier.event.is_set()


def test_metric_summary_is_empty_safe():
    assert metric_summary([])["p50"] is None
