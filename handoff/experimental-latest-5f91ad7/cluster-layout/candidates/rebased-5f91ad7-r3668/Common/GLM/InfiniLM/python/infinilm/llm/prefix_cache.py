"""Prefix-cache block hashing utilities."""

from collections.abc import Sequence

import numpy as np
import xxhash

BlockHash = bytes
EMPTY_BLOCK_HASH = b""

_HASH_SCHEMA = b"InfiniLM-prefix-cache-v1\0"
_EMPTY_PARENT_MARKER = b"\0"


def hash_block_tokens(
    token_ids: Sequence[int], parent_hash: BlockHash = EMPTY_BLOCK_HASH
) -> BlockHash:
    """Hash one full token block, chained to its parent prefix."""
    if parent_hash and len(parent_hash) != 16:
        raise ValueError("parent_hash must be a 128-bit digest")

    token_array = np.asarray(token_ids, dtype="<i4")
    hasher = xxhash.xxh3_128()
    hasher.update(_HASH_SCHEMA)
    hasher.update(parent_hash or _EMPTY_PARENT_MARKER)
    hasher.update(len(token_array).to_bytes(4, "little"))
    hasher.update(token_array.tobytes())
    return hasher.digest()
