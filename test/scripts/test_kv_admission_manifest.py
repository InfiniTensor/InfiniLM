import json
from pathlib import Path

import pytest

from scripts.kv_admission_manifest import load_manifest


def write_manifest(path: Path, requests):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "configuration": {
                    "model": "test-model",
                    "block_size": 16,
                    "num_blocks": 32,
                    "seed": 7,
                },
                "requests": requests,
            }
        ),
        encoding="utf-8",
    )


def test_manifest_preserves_request_order_and_wave_partition(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        [
            {
                "request_id": "wave1-0000",
                "wave": "wave1",
                "prompt": "alpha",
                "prompt_token_ids": [1, 2, 3],
                "max_tokens": 8,
            },
            {
                "request_id": "wave2-0000",
                "wave": "wave2",
                "prompt": "beta",
                "prompt_token_ids": [4, 5],
                "max_tokens": 4,
            },
        ],
    )

    manifest = load_manifest(path)

    assert [item.request_id for item in manifest.requests] == [
        "wave1-0000",
        "wave2-0000",
    ]
    assert manifest.wave1[0].prompt_tokens == 3
    assert manifest.wave2[0].max_tokens == 4
    assert len(manifest.sha256) == 64


def test_manifest_rejects_declared_length_mismatch(tmp_path):
    path = tmp_path / "manifest.json"
    write_manifest(
        path,
        [
            {
                "request_id": "wave1-0000",
                "wave": "wave1",
                "prompt": "alpha",
                "prompt_token_ids": [1, 2, 3],
                "prompt_tokens": 99,
                "max_tokens": 8,
            },
            {
                "request_id": "wave2-0000",
                "wave": "wave2",
                "prompt": "beta",
                "prompt_token_ids": [4],
                "max_tokens": 4,
            },
        ],
    )

    with pytest.raises(ValueError, match="does not match"):
        load_manifest(path)
