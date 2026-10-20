from __future__ import annotations

import json
import struct
from pathlib import Path


def make_model_dir(root: Path, *, payload: bytes = b"0123456789abcdef") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text('{"model_type":"unit"}\n', encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (root / "model.safetensors").write_bytes(payload)
    nested = root / "nested"
    nested.mkdir()
    (nested / "extra.bin").write_bytes(b"extra")
    (root / ".iloader-cache").mkdir()
    (root / ".iloader-cache" / "ignored.bin").write_bytes(b"ignored")
    (root / "debug.log").write_text("ignored\n", encoding="utf-8")
    return root


def write_safetensors(path: Path, tensors: dict[str, dict], payload_size: int) -> None:
    header = json.dumps(tensors, separators=(",", ":")).encode("utf-8")
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(bytes(payload_size))
