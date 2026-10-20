"""Private, read-only safetensors byte views for checkpoint copies.

This is deliberately not a general tensor loader. ``tensor_bytes`` returns a
NumPy integer bit-view whose base keeps the read-only mmap alive; the caller
must supply the checked safetensors dtype when borrowing it with from_blob.
"""

import json
import mmap
import os
import sys
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

_MAX_HEADER_BYTES = 100_000_000  # The safetensors format's header limit.
_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
    "F8_E8M0": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
}
_BITS = {1: np.dtype("u1"), 2: np.dtype("<u2"), 4: np.dtype("<u4"), 8: np.dtype("<u8")}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate safetensors header key: {key!r}")
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError(f"non-JSON safetensors constant: {value}")


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: tuple[int, ...]
    begin: int
    end: int


class SafeTensorsMmap:
    """Validate one shard and expose only borrowed, read-only bit-views.

    Keep the returned arrays alive until the consuming H2D copy completes.
    Closing the reader does not invalidate outstanding borrowed arrays.
    """

    def __init__(self, path: os.PathLike | str):
        self._mapping = None
        with open(path, "rb") as file:
            if os.fstat(file.fileno()).st_size < 10:
                raise ValueError("safetensors file is too short")
            mapping = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            tensors, metadata, data_start = self._validate(mapping)
        except BaseException:
            mapping.close()
            raise
        self._mapping = mapping
        self.tensors = MappingProxyType(tensors)
        self.metadata = None if metadata is None else MappingProxyType(metadata)
        self._data_start = data_start

    @staticmethod
    def _validate(mapping):
        header_size = int.from_bytes(mapping[:8], "little")
        if not 2 <= header_size <= _MAX_HEADER_BYTES:
            raise ValueError("invalid safetensors header size")
        data_start = 8 + header_size
        if data_start > len(mapping):
            raise ValueError("safetensors header exceeds file size")
        header = mapping[8:data_start]
        if header[:1] != b"{":
            raise ValueError("safetensors header must begin with '{'")
        try:
            decoded = header.decode("utf-8")
            decoder = json.JSONDecoder(
                object_pairs_hook=_unique_object, parse_constant=_bad_constant
            )
            entries, consumed = decoder.raw_decode(decoded)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid safetensors JSON header") from exc
        if decoded[consumed:].strip(" ") or not isinstance(entries, dict):
            raise ValueError("invalid safetensors header or padding")

        metadata = None
        if "__metadata__" in entries:
            metadata = entries.pop("__metadata__")
            if not isinstance(metadata, dict) or any(
                not isinstance(key, str) or not isinstance(value, str)
                for key, value in metadata.items()
            ):
                raise ValueError("safetensors metadata must map strings to strings")
        try:
            for key, value in (metadata or {}).items():
                key.encode("utf-8")
                value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("invalid Unicode in safetensors metadata") from exc

        data_size = len(mapping) - data_start
        tensors = {}
        intervals = []
        for name, entry in entries.items():
            if (
                not isinstance(name, str)
                or not isinstance(entry, dict)
                or set(entry) != {"dtype", "shape", "data_offsets"}
            ):
                raise ValueError(f"invalid safetensors entry: {name!r}")
            try:
                name.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError(
                    f"invalid Unicode in safetensors name: {name!r}"
                ) from exc
            dtype, shape, offsets = (
                entry["dtype"],
                entry["shape"],
                entry["data_offsets"],
            )
            if not isinstance(dtype, str) or dtype not in _ITEMSIZE:
                raise ValueError(
                    f"unsupported safetensors dtype for {name!r}: {dtype!r}"
                )
            if not isinstance(shape, list) or any(
                type(dim) is not int or dim < 0 or dim > sys.maxsize for dim in shape
            ):
                raise ValueError(f"invalid safetensors shape for {name!r}")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    type(value) is not int or value < 0 or value > data_size
                    for value in offsets
                )
            ):
                raise ValueError(f"invalid safetensors offsets for {name!r}")
            begin, end = offsets
            if end < begin:
                raise ValueError(f"reversed safetensors offsets for {name!r}")
            count = 1
            for dim in shape:
                if dim and count > sys.maxsize // dim:
                    raise ValueError(f"safetensors shape overflows for {name!r}")
                count *= dim
            if (
                count > sys.maxsize // _ITEMSIZE[dtype]
                or end - begin != count * _ITEMSIZE[dtype]
            ):
                raise ValueError(f"safetensors byte size mismatch for {name!r}")
            tensors[name] = TensorInfo(dtype, tuple(shape), begin, end)
            if begin != end:
                intervals.append((begin, end))

        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError("safetensors data has a hole or overlap")
            cursor = end
        if cursor != data_size:
            raise ValueError("safetensors data is not entirely indexed")
        return tensors, metadata, data_start

    def tensor_bytes(self, name: str) -> np.ndarray:
        """Return a C-contiguous bit-view, including BF16 as uint16 bits."""
        if self._mapping is None:
            raise ValueError("safetensors mmap is closed")
        info = self.tensors[name]
        itemsize = _ITEMSIZE[info.dtype]
        array = np.frombuffer(
            self._mapping,
            dtype=_BITS[itemsize],
            count=(info.end - info.begin) // itemsize,
            offset=self._data_start + info.begin,
        )
        return array.reshape(info.shape)

    def close(self) -> None:
        if self._mapping is not None:
            mapping = self._mapping
            self._mapping = None
            try:
                mapping.close()
            except BufferError:
                # NumPy views own exports of this mmap. Detaching the reader
                # leaves the mapping alive only until those views are gone.
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
