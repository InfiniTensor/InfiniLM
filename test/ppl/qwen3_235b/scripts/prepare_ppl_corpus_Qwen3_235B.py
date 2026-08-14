#!/usr/bin/env python3
"""将本地纯文本固化为 Qwen3_235B PPL 测试使用的 token manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import operator
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from _ppl_common import (
    CORPUS_SCHEMA,
    canonical_json_bytes,
    canonical_token_ids_sha256,
    write_token_ids_npy,
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def _tokenizer_fingerprint(tokenizer: Any) -> tuple[str, str]:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None and hasattr(backend, "to_str"):
        try:
            backend_payload: object = json.loads(backend.to_str())
        except (TypeError, ValueError, json.JSONDecodeError):
            backend_payload = backend.to_str()
        method = "backend_tokenizer+special_tokens/v1"
        semantics = {
            "backend_tokenizer": backend_payload,
            "special_tokens_map": _jsonable(
                getattr(tokenizer, "special_tokens_map", {})
            ),
        }
    else:
        if not hasattr(tokenizer, "get_vocab"):
            raise RuntimeError("tokenizer 既没有 backend_tokenizer，也没有 get_vocab()")
        method = "vocab+init_kwargs+special_tokens/v1"
        semantics = {
            "vocab": tokenizer.get_vocab(),
            "init_kwargs": _jsonable(getattr(tokenizer, "init_kwargs", {})),
            "special_tokens_map": _jsonable(
                getattr(tokenizer, "special_tokens_map", {})
            ),
        }
    return hashlib.sha256(canonical_json_bytes(semantics)).hexdigest(), method


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="本地 UTF-8 WikiText/raw text 文件，可按顺序提供多个文件",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="本地 Qwen3_235B 模型或 tokenizer 目录",
    )
    parser.add_argument("--output", required=True, type=Path, help="输出 JSON manifest")
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="仅保留开头 N 个 token；省略时保留全部 token",
    )
    parser.add_argument(
        "--document-separator",
        default="\n\n",
        help=r"多个输入文件之间的分隔符，默认 '\n\n'",
    )
    parser.add_argument(
        "--storage",
        choices=("inline", "npy"),
        default="inline",
        help="token IDs 内联到 JSON（默认），或写入相对路径 .npy 文件",
    )
    parser.add_argument(
        "--token-ids-file",
        type=Path,
        help="--storage=npy 时的相对路径；默认 <manifest>.tokens.npy",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="允许 Transformers 访问网络；默认只读取本地文件/缓存",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="传给 AutoTokenizer；Qwen3 官方 tokenizer 通常不需要",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出")
    args = parser.parse_args(argv)

    if args.max_tokens is not None and args.max_tokens < 2:
        parser.error("--max-tokens 必须至少为 2")
    if args.token_ids_file is not None and args.storage != "npy":
        parser.error("--token-ids-file 只能与 --storage=npy 一起使用")
    if len({path.name for path in args.input}) != len(args.input):
        parser.error("输入文件名不能重复，否则 manifest 无法稳定区分来源")
    return args


def _read_sources(
    paths: Sequence[Path], separator: str
) -> tuple[str, list[dict[str, object]]]:
    documents: list[str] = []
    files: list[dict[str, object]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"输入文件不存在：{path}")
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"输入文件不是有效 UTF-8：{path}: {error}") from error
        documents.append(text)
        files.append(
            {
                "name": path.name,
                "byte_count": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return separator.join(documents), files


def _load_tokenizer(identifier: str, allow_download: bool, trust_remote_code: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("缺少 transformers，无法加载 tokenizer") from error
    return AutoTokenizer.from_pretrained(
        identifier,
        local_files_only=not allow_download,
        trust_remote_code=trust_remote_code,
    )


def _encode(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    raw_ids = encoded["input_ids"]
    if not isinstance(raw_ids, (list, tuple)) or (
        raw_ids and isinstance(raw_ids[0], (list, tuple))
    ):
        raise RuntimeError("tokenizer 必须为单条文本返回一维 input_ids")
    token_ids: list[int] = []
    for index, value in enumerate(raw_ids):
        if isinstance(value, bool):
            raise RuntimeError(f"input_ids[{index}] 不是有效整数")
        try:
            token = operator.index(value)
        except TypeError as error:
            raise RuntimeError(f"input_ids[{index}] 不是有效整数") from error
        if token < 0:
            raise RuntimeError(f"input_ids[{index}] 不是非负整数：{value!r}")
        token_ids.append(token)
    if len(token_ids) < 2:
        raise RuntimeError("语料 token 数不足 2，无法计算 causal LM PPL")
    return token_ids


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _npy_relative_path(output: Path, requested: Path | None) -> Path:
    relative = requested or Path(f"{output.stem}.tokens.npy")
    if relative.is_absolute() or ".." in relative.parts or relative.name in {"", "."}:
        raise ValueError("--token-ids-file 必须是 manifest 目录内的安全相对路径")
    if relative.suffix != ".npy":
        raise ValueError("--token-ids-file 必须以 .npy 结尾")
    return relative


def _write_npy(path: Path, token_ids: Sequence[int], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"token 文件已存在（可加 --overwrite）：{path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    try:
        write_token_ids_npy(temporary, token_ids)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_manifest(
    *,
    source_text: str,
    source_files: list[dict[str, object]],
    separator: str,
    tokenizer: Any,
    tokenizer_label: str,
    token_ids: list[int],
    original_token_count: int,
    max_tokens: int | None,
    storage: str,
    token_ids_file: str | None,
) -> dict[str, object]:
    tokenizer_hash, fingerprint_method = _tokenizer_fingerprint(tokenizer)
    token_hash = canonical_token_ids_sha256(token_ids)
    payload: dict[str, object] = {
        "schema": CORPUS_SCHEMA,
        "source_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        "tokenizer_sha256": tokenizer_hash,
        "token_count": len(token_ids),
        "token_ids_sha256": token_hash,
        "source": {
            "encoding": "utf-8",
            "document_separator": separator,
            "file_count": len(source_files),
            "files": source_files,
        },
        "tokenizer": {
            "name": Path(tokenizer_label.rstrip("/")).name or tokenizer_label,
            "class": tokenizer.__class__.__name__,
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
            "fingerprint_method": fingerprint_method,
        },
        "tokenization": {
            "add_special_tokens": False,
            "original_token_count": original_token_count,
            "max_tokens": max_tokens,
            "truncated": len(token_ids) != original_token_count,
        },
    }
    if storage == "inline":
        payload["token_ids"] = token_ids
    else:
        if token_ids_file is None:
            raise ValueError("npy storage 缺少 token_ids_file")
        payload["token_ids_file"] = token_ids_file
        payload["token_ids_dtype"] = "int64"
    payload["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.output.exists() and not args.overwrite:
            raise FileExistsError(f"输出已存在（可加 --overwrite）：{args.output}")
        source_text, source_files = _read_sources(args.input, args.document_separator)
        tokenizer = _load_tokenizer(
            args.tokenizer, args.allow_download, args.trust_remote_code
        )
        all_token_ids = _encode(tokenizer, source_text)
        original_token_count = len(all_token_ids)
        token_ids = (
            all_token_ids[: args.max_tokens]
            if args.max_tokens is not None
            else all_token_ids
        )

        relative_npy: Path | None = None
        if args.storage == "npy":
            relative_npy = _npy_relative_path(args.output, args.token_ids_file)

        manifest = build_manifest(
            source_text=source_text,
            source_files=source_files,
            separator=args.document_separator,
            tokenizer=tokenizer,
            tokenizer_label=args.tokenizer,
            token_ids=token_ids,
            original_token_count=original_token_count,
            max_tokens=args.max_tokens,
            storage=args.storage,
            token_ids_file=str(relative_npy) if relative_npy is not None else None,
        )
        if relative_npy is not None:
            _write_npy(args.output.parent / relative_npy, token_ids, args.overwrite)
        _atomic_json(args.output, manifest)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    print(f"PPL 语料已固化：{args.output}")
    print(f"Token 数：{manifest['token_count']}")
    print(f"Token SHA256：{manifest['token_ids_sha256']}")
    print(f"Manifest SHA256：{manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
