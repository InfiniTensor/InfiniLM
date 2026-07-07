"""Lightweight multimodal feature metadata used by prefix caching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator


@dataclass(frozen=True)
class MMPlaceholderRange:
    """Placeholder token range for a multimodal input."""

    offset: int
    length: int
    is_embed: Any | None = None

    @property
    def end(self) -> int:
        return self.offset + self.length

    def overlap(self, start: int, end: int) -> bool:
        return end > self.offset and start < self.end

    def overlaps(self, start: int, end: int) -> bool:
        return self.overlap(start, end)


@dataclass(frozen=True)
class MMFeaturePart:
    """Model-specific part of a multimodal feature."""

    token_start: int
    token_end: int
    embed_start: int
    embed_end: int
    item_index: int
    part_index: int
    data_index: int | None = None

    @property
    def token_length(self) -> int:
        return self.token_end - self.token_start

    def overlap(self, start: int, end: int) -> bool:
        return end > self.token_start and start < self.token_end

    def overlaps(self, start: int, end: int) -> bool:
        return self.overlap(start, end)


@dataclass(frozen=True)
class MMFeature:
    """A cache-addressable multimodal input feature."""

    modality: str
    identifier: str
    position: MMPlaceholderRange
    mm_hash: str | None = None
    data: Any | None = None
    parts: tuple[MMFeaturePart, ...] = field(default_factory=tuple)


def iter_mm_parts(
    mm_features: Iterable[MMFeature],
) -> Iterator[tuple[MMFeature, MMFeaturePart]]:
    for feature in mm_features:
        for part in feature.parts:
            yield feature, part
