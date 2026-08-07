#!/usr/bin/env python3
"""Shared, deterministic corpus and sliding-window helpers for true PPL tests."""

from __future__ import annotations

import ast
import array
import hashlib
import json
import operator
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


CORPUS_SCHEMA = "qw235_ppl_token_ids_v1"
SCORING_METHOD = (
    "sliding_window_shifted_cross_entropy_fp32_compute_fp64_accumulation"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _canonical_int_sequence_sha256(values: Iterable[int], label: str) -> str:
    """Hash an integer sequence exactly like compact JSON ``[1,2,3]``."""
    digest = hashlib.sha256()
    digest.update(b"[")
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"{label}[{index}] 不是非负整数")
        try:
            parsed = operator.index(value)
        except TypeError as error:
            raise ValueError(f"{label}[{index}] 不是非负整数") from error
        if parsed < 0:
            raise ValueError(f"{label}[{index}] 不是非负整数：{value!r}")
        if index:
            digest.update(b",")
        digest.update(str(parsed).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def canonical_token_ids_sha256(token_ids: Iterable[int]) -> str:
    return _canonical_int_sequence_sha256(token_ids, "token_ids")


def canonical_indices_sha256(indices: Iterable[int]) -> str:
    return _canonical_int_sequence_sha256(indices, "indices")


@dataclass(frozen=True)
class PplCorpusManifest:
    path: Path
    payload: dict[str, Any]
    token_ids: tuple[int, ...]
    manifest_sha256: str
    token_ids_sha256: str

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class SlidingWindow:
    """One causal-LM window using half-open global token index ranges.

    ``token_start:token_end`` is model input. Targets in
    ``score_start:score_end`` are scored. ``prediction_*`` select the matching
    logits before the causal shift, while ``target_*`` select labels locally.
    """

    index: int
    token_start: int
    token_end: int
    score_start: int
    score_end: int
    token_ids: tuple[int, ...]

    @property
    def scored_token_count(self) -> int:
        return self.score_end - self.score_start

    @property
    def prediction_start(self) -> int:
        return self.score_start - self.token_start - 1

    @property
    def prediction_end(self) -> int:
        return self.score_end - self.token_start - 1

    @property
    def target_start(self) -> int:
        return self.score_start - self.token_start

    @property
    def target_end(self) -> int:
        return self.score_end - self.token_start


def _required_positive_int(payload: dict[str, Any], key: str, path: Path) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} 的 {key} 必须是正整数")
    parsed = value
    if parsed <= 0:
        raise ValueError(f"{path} 的 {key} 必须是正整数")
    return parsed


def _required_sha(payload: dict[str, Any], key: str, path: Path) -> str:
    value = str(payload.get(key, "")).lower()
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"{path} 的 {key} 不是有效 SHA256")
    return value


def _load_npy(manifest_path: Path, relative_name: object) -> list[int]:
    relative = Path(str(relative_name))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{manifest_path} 的 token_ids_file 必须是安全相对路径")
    base = manifest_path.parent.resolve()
    token_path = (base / relative).resolve()
    try:
        token_path.relative_to(base)
    except ValueError as error:
        raise ValueError(f"token_ids_file 越出 manifest 目录：{relative}") from error
    try:
        import numpy as np
    except ImportError:
        return _load_int64_npy_without_numpy(token_path)
    try:
        array = np.load(token_path, allow_pickle=False)
    except FileNotFoundError:
        raise ValueError(f"token_ids_file 不存在：{token_path}") from None
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{token_path} 必须是一维整数 .npy 数组")
    return [int(value) for value in array.tolist()]


def _load_int64_npy_without_numpy(path: Path) -> list[int]:
    try:
        with path.open("rb") as handle:
            if handle.read(6) != b"\x93NUMPY":
                raise ValueError(f"{path} 不是有效 .npy 文件")
            version = handle.read(2)
            if version == b"\x01\x00":
                header_length = struct.unpack("<H", handle.read(2))[0]
            elif version in {b"\x02\x00", b"\x03\x00"}:
                header_length = struct.unpack("<I", handle.read(4))[0]
            else:
                raise ValueError(f"{path} 使用不支持的 .npy 版本 {tuple(version)}")
            try:
                header = ast.literal_eval(handle.read(header_length).decode("latin1"))
            except (SyntaxError, ValueError, UnicodeDecodeError) as error:
                raise ValueError(f"{path} 的 .npy header 无效") from error
            if not isinstance(header, dict):
                raise ValueError(f"{path} 的 .npy header 不是字典")
            shape = header.get("shape")
            if (
                header.get("descr") != "<i8"
                or header.get("fortran_order") is not False
                or not isinstance(shape, tuple)
                or len(shape) != 1
                or not isinstance(shape[0], int)
                or shape[0] < 0
            ):
                raise ValueError(
                    f"{path} 无 NumPy 后备读取仅支持一维 little-endian int64"
                )
            raw = handle.read()
    except FileNotFoundError:
        raise ValueError(f"token_ids_file 不存在：{path}") from None
    expected_bytes = shape[0] * 8
    if len(raw) != expected_bytes:
        raise ValueError(
            f"{path} 数据长度错误：expected={expected_bytes}, actual={len(raw)}"
        )
    return [value[0] for value in struct.iter_unpack("<q", raw)]


def write_token_ids_npy(path: str | Path, token_ids: Sequence[int]) -> None:
    """Write a portable NumPy v1.0, one-dimensional little-endian int64 file."""
    output = Path(path)
    values = list(token_ids)
    # Validate before creating a partial file.
    canonical_token_ids_sha256(values)
    header_text = repr(
        {"descr": "<i8", "fortran_order": False, "shape": (len(values),)}
    )
    header_without_padding = header_text.encode("latin1")
    padding = (-(10 + len(header_without_padding) + 1)) % 16
    header = header_without_padding + (b" " * padding) + b"\n"
    if len(header) > 65535:
        raise ValueError(".npy header 超过 v1.0 长度限制")
    output.parent.mkdir(parents=True, exist_ok=True)
    packed = array.array("q", (int(value) for value in values))
    if packed.itemsize != 8:
        raise RuntimeError("当前 Python 平台的 signed long long 不是 64 bit")
    if sys.byteorder != "little":
        packed.byteswap()
    with output.open("wb") as handle:
        handle.write(b"\x93NUMPY")
        handle.write(b"\x01\x00")
        handle.write(struct.pack("<H", len(header)))
        handle.write(header)
        packed.tofile(handle)


def load_manifest(path: str | Path) -> PplCorpusManifest:
    """Load and fully verify an inline or relative-``.npy`` corpus manifest."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"PPL manifest 不存在：{manifest_path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"PPL manifest JSON 无效：{manifest_path}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"PPL manifest 必须是 JSON 对象：{manifest_path}")
    if payload.get("schema") != CORPUS_SCHEMA:
        raise ValueError(
            f"{manifest_path} schema 必须为 {CORPUS_SCHEMA!r}，"
            f"实际为 {payload.get('schema')!r}"
        )
    manifest_hash = _required_sha(payload, "manifest_sha256", manifest_path)
    semantic_payload = dict(payload)
    semantic_payload.pop("manifest_sha256", None)
    calculated_manifest_hash = hashlib.sha256(
        canonical_json_bytes(semantic_payload)
    ).hexdigest()
    if manifest_hash != calculated_manifest_hash:
        raise ValueError(f"{manifest_path} 的 manifest_sha256 校验失败")
    for key in ("source_sha256", "tokenizer_sha256"):
        _required_sha(payload, key, manifest_path)

    has_inline = "token_ids" in payload
    has_file = "token_ids_file" in payload
    if has_inline == has_file:
        raise ValueError(
            f"{manifest_path} 必须且只能包含 token_ids 或 token_ids_file 之一"
        )
    if has_inline:
        raw_ids = payload["token_ids"]
        if not isinstance(raw_ids, list):
            raise ValueError(f"{manifest_path} 的 token_ids 必须是数组")
        token_ids = list(raw_ids)
    else:
        token_ids = _load_npy(manifest_path, payload["token_ids_file"])

    # The canonical hash validates type, integrality, sign, order and contents.
    calculated_token_hash = canonical_token_ids_sha256(token_ids)
    expected_token_hash = _required_sha(
        payload, "token_ids_sha256", manifest_path
    )
    if calculated_token_hash != expected_token_hash:
        raise ValueError(f"{manifest_path} 的 token_ids_sha256 校验失败")
    token_count = _required_positive_int(payload, "token_count", manifest_path)
    if token_count != len(token_ids) or token_count < 2:
        raise ValueError(
            f"{manifest_path} token_count={token_count}，实际 token 数={len(token_ids)}"
        )
    return PplCorpusManifest(
        path=manifest_path,
        payload=payload,
        token_ids=tuple(int(value) for value in token_ids),
        manifest_sha256=manifest_hash,
        token_ids_sha256=expected_token_hash,
    )


def iter_sliding_windows(
    token_ids: Sequence[int],
    window_size: int,
    stride: int,
    max_scored_tokens: int | None = None,
) -> Iterator[SlidingWindow]:
    """Yield windows that score global token indices ``1..N-1`` exactly once.

    The first window scores ``1:end``. Every later window scores only
    ``previous_end:end``; overlapped prefix tokens provide context but are not
    counted again. ``stride`` must be smaller than ``window_size`` so the first
    new target in every later window retains its immediately preceding token.
    """
    if isinstance(window_size, bool) or not isinstance(window_size, int):
        raise ValueError("window_size 必须是整数")
    if isinstance(stride, bool) or not isinstance(stride, int):
        raise ValueError("stride 必须是整数")
    if window_size < 2:
        raise ValueError("window_size 必须至少为 2")
    if stride < 1 or stride >= window_size:
        raise ValueError("stride 必须满足 1 <= stride < window_size")
    if len(token_ids) < 2:
        raise ValueError("至少需要 2 个 token 才能计算 PPL")
    if max_scored_tokens is not None:
        if (
            isinstance(max_scored_tokens, bool)
            or not isinstance(max_scored_tokens, int)
            or max_scored_tokens < 1
        ):
            raise ValueError("max_scored_tokens 必须是正整数")

    score_limit = len(token_ids)
    if max_scored_tokens is not None:
        score_limit = min(score_limit, 1 + max_scored_tokens)
    previous_end = 1
    index = 0
    while previous_end < score_limit:
        if index == 0:
            token_end = min(score_limit, window_size)
            token_start = 0
            score_start = 1
        else:
            token_end = min(score_limit, previous_end + stride)
            token_start = max(0, token_end - window_size)
            score_start = previous_end
        window = SlidingWindow(
            index=index,
            token_start=token_start,
            token_end=token_end,
            score_start=score_start,
            score_end=token_end,
            token_ids=tuple(int(value) for value in token_ids[token_start:token_end]),
        )
        if window.prediction_start < 0 or window.prediction_end > len(window.token_ids):
            raise AssertionError("内部错误：滑窗缺少 causal predecessor")
        yield window
        previous_end = token_end
        index += 1
