"""Structural descriptions of speculative draft models.

A checkpoint is accepted as a speculative draft when this module can describe
it: the checkpoint publishes draft-layer weights under a family-specific key
prefix (criterion C1), the number of draft depths is derivable from metadata or
from the registration itself (C2), and the draft block is a serial
"target hidden -> next-token logits" forward whose layer kinds this build can
compose (C3/C4). The speculative runner and the weight loader hold no
family-specific judgement of their own: they ask this registry what a
checkpoint is and how to build it.

Adding a family therefore means adding one `DraftModelSpec` plus one
`register_draft_model_spec` call below. Families whose draft block this build
cannot run yet are described here as well: they resolve to an actionable
construction error naming what is missing, instead of failing silently or being
reported as "not a draft checkpoint". The user-facing walkthrough of the
criteria and of the fields lives in MODELS.md, "Adding a new MTP draft model".

The same descriptions answer for the checkpoint's **target** side: a family
whose head is embedded in the target checkpoint states how its own draft
tensors are recognised, and loading that checkpoint as a target drops exactly
those tensors, so the target side needs no entry of its own in the loader's
remapper table.
"""

import json
import logging
import os
import re
import struct
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


class UnsupportedDraftError(RuntimeError):
    """A checkpoint is not a draft that this build can construct."""


# The guide section that defines the acceptance criteria a description has to
# satisfy. Error messages point at the section by name rather than at a
# criterion number: the numbers are easy to renumber and the mistakes that
# mattered were references to numbers the guide never defined.
CRITERIA_SECTION = 'the acceptance criteria in "Adding a new MTP draft model"'


class DraftLayerKind(str, Enum):
    """Attention/MLP combination a draft block needs."""

    FULL_ATTENTION = "full_attention"
    LINEAR_ATTENTION = "linear_attention"
    HYBRID = "hybrid"
    SLIDING_ATTENTION = "sliding_attention"
    MOE = "moe"
    MLA_ATTENTION = "mla_attention"


class FusionMode(str, Enum):
    """How the token embedding and the target hidden state are combined."""

    CONCAT_PROJECTION = "concat_projection"
    ADD_PROJECTION = "add_projection"
    INDEPENDENT_PROJECTION_SHARED_RESIDUAL = "independent_projection_shared_residual"


class ConcatOrder(str, Enum):
    """Order of the two streams when the fusion concatenates them."""

    EMBEDDING_FIRST = "embedding_first"
    HIDDEN_FIRST = "hidden_first"
    UNSPECIFIED = "unspecified"


class EmbeddingAtPositionZero(str, Enum):
    """What the embedding stream carries at the first drafted position."""

    ZEROED = "zeroed"
    PREVIOUS_TOKEN = "previous_token"
    UNSPECIFIED = "unspecified"


class RecycleHidden(str, Enum):
    """Tensor fed back as the next draft step's hidden input."""

    PRE_FINAL_NORM = "pre_final_norm"
    POST_FINAL_NORM = "post_final_norm"


class EmbeddingSharing(str, Enum):
    """Where the draft's token embedding and output head come from."""

    SHARED_WITH_TARGET = "shared_with_target"
    TARGET_EMBEDDING_AND_HEAD = "target_embedding_and_head"
    PER_DEPTH = "per_depth"
    DRAFT_CHECKPOINT = "draft_checkpoint"


class LayerSource(str, Enum):
    """Where the draft block's decoder layer comes from."""

    REUSE_TARGET = "reuse_target"
    FAMILY_SPECIFIC = "family_specific"


class PositionIdLayout(str, Enum):
    """Shape of the position ids the draft engine expects."""

    STANDARD = "standard"
    MROPE_TEXT = "mrope_text"


@dataclass(frozen=True)
class DraftWeightMap:
    """How a family's published draft keys are recognised and renamed.

    Three patterns, one job each:

    - ``family_keys`` recognises that a checkpoint carries this family's draft
      head at all. It matches the family's own key namespaces (its fusion and
      norm tensors, its draft-layer namespace), never a generic layout such as
      ``model.layers.<N>.`` that every checkpoint has. Recognition is what
      decides "which family is this", so it stays narrow.
    - ``layer_key_pattern``, when the family publishes one block per depth,
      matches the keys of a single draft layer and exposes its index in the
      ``depth`` group. It drives the published-depth count, which is
      cross-checked against the config so a counting key that disagrees with
      the weights is reported rather than trusted. Families whose head is not
      split per depth leave it empty and declare their depth instead.
    - ``key_pattern`` selects the tensors to rename; its optional ``depth``
      group is anchored on the target config field named by
      ``depth_index_key`` (DeepSeek-V3 publishes its MTP block at layer
      ``num_hidden_layers``, which still maps onto the draft's own depth 0) and
      its ``rest`` group becomes the canonical key.

    ``renames`` rewrites canonical-name fragments in order (families that call
    the fusion projection ``input_proj`` rather than ``fc``, for instance).
    ``zero_centered_keys`` / ``zero_centered_norms`` list the weights stored
    around zero instead of around one, by exact canonical key and by suffix.
    ``embedding_keys`` / ``head_keys`` are the target side tensors the draft
    shares.
    """

    family_keys: str
    key_pattern: str
    layer_key_pattern: Optional[str] = None
    canonical_prefix: str = "model."
    depth_index_key: Optional[str] = None
    renames: tuple[tuple[str, str], ...] = ()
    zero_centered_keys: tuple[str, ...] = ()
    zero_centered_norms: tuple[str, ...] = ()
    embedding_keys: tuple[str, ...] = ()
    head_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class DraftModelSpec:
    """One family's draft-head description.

    The fields after the packaging block describe the draft block itself. What
    the description states is what the family's checkpoint needs; the draft
    model registered under ``draft_model_type`` is what executes it, so a
    description and its draft model must agree. Semantics this build does not
    execute are still recorded rather than dropped: the next family
    integration, and a reader of MODELS.md, needs them to judge the family.

    A checkpoint is attributed to this family when its ``model_type`` is listed
    in ``target_model_types`` and it publishes ``family_keys``; neither half
    alone is treated as an identification.
    """

    family: str
    draft_model_type: Optional[str] = None
    target_model_types: tuple[str, ...] = ()
    embedded: bool = False
    depth_keys: tuple[str, ...] = ()
    declared_depth: Optional[int] = None
    runtime_depth: Optional[int] = None
    runtime_depth_keys: tuple[str, ...] = ()
    shared_layer_depths: tuple[int, ...] = ()
    layer_config_key: Optional[str] = "layer_types"
    layer_kinds: tuple[DraftLayerKind, ...] = ()
    fusion: FusionMode = FusionMode.CONCAT_PROJECTION
    concat_order: ConcatOrder = ConcatOrder.EMBEDDING_FIRST
    embedding_at_position_zero: EmbeddingAtPositionZero = (
        EmbeddingAtPositionZero.UNSPECIFIED
    )
    recycle_hidden: RecycleHidden = RecycleHidden.POST_FINAL_NORM
    hidden_streams: int = 1
    chain_causal: bool = True
    embedding_sharing: EmbeddingSharing = EmbeddingSharing.SHARED_WITH_TARGET
    layer_source: LayerSource = LayerSource.REUSE_TARGET
    position_ids: PositionIdLayout = PositionIdLayout.STANDARD
    weight_map: Optional[DraftWeightMap] = None
    rejected_config_flags: tuple[tuple[str, object, str], ...] = ()
    unimplemented: tuple[str, ...] = ()

    @property
    def is_available(self) -> bool:
        """Whether this build can construct the family's draft block."""
        return not self.unimplemented


@dataclass(frozen=True)
class DraftCheckpoint:
    """A draft checkpoint resolved against a description."""

    spec: DraftModelSpec
    engine_path: str
    published_depth: int
    depth_source: str
    layer_kinds: tuple[DraftLayerKind, ...]


DRAFT_MODEL_SPECS: dict[str, DraftModelSpec] = {}
_DERIVED_REMAPPERS: dict[str, Callable] = {}
_DERIVED_TARGET_REMAPPERS: dict[str, Callable] = {}


def register_draft_model_spec(spec: DraftModelSpec) -> DraftModelSpec:
    """Register one family description under its family name."""
    DRAFT_MODEL_SPECS[spec.family] = spec
    _DERIVED_REMAPPERS.pop(spec.family, None)
    _DERIVED_TARGET_REMAPPERS.pop(spec.family, None)
    return spec


def list_draft_model_specs() -> list[DraftModelSpec]:
    """All registered descriptions, in registration order."""
    return list(DRAFT_MODEL_SPECS.values())


def get_draft_model_spec(model_type: str) -> Optional[DraftModelSpec]:
    """Return the description a draft engine's ``model_type`` belongs to.

    A checkpoint that embeds its draft weights keeps the description that
    resolved it, so this lookup decides for standalone draft checkpoints; when
    several descriptions share a draft model type, the first registered wins.
    """
    for spec in DRAFT_MODEL_SPECS.values():
        if spec.draft_model_type == model_type or spec.family == model_type:
            return spec
    return None


def _read_config(model_path: str) -> Optional[dict]:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r") as f:
        return json.load(f)


def _config_scopes(hf_config: dict) -> list[dict]:
    """Config views a family field may live in: the top level and text_config."""
    scopes = [hf_config]
    text_config = hf_config.get("text_config")
    if isinstance(text_config, dict):
        scopes.append(text_config)
    return scopes


def _published_keys(model_path: str) -> Optional[set[str]]:
    """Tensor names the checkpoint publishes, without reading tensor data."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return set(json.load(f).get("weight_map", {}))
    shards = sorted(
        name for name in os.listdir(model_path) if name.endswith(".safetensors")
    )
    if not shards:
        return None
    keys: set[str] = set()
    for name in shards:
        keys.update(_safetensors_header_keys(os.path.join(model_path, name)))
    return keys


def _safetensors_header_keys(path: str) -> list[str]:
    # The header is a little-endian u64 length followed by a JSON object; only
    # the header is read, so this stays cheap for multi-gigabyte shards.
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    return [key for key in header if key != "__metadata__"]


def _literal_prefix(pattern: str) -> str:
    """Literal part of a key pattern, up to its first regex metacharacter."""
    prefix = []
    index = 0
    if pattern.startswith("^"):
        index = 1
    while index < len(pattern):
        char = pattern[index]
        if char == "\\" and index + 1 < len(pattern):
            prefix.append(pattern[index + 1])
            index += 2
            continue
        if char in ".^$*+?()[]{}|":
            break
        prefix.append(char)
        index += 1
    return "".join(prefix)


def _int_or_none(value: object) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _config_depth(hf_config: dict, spec: DraftModelSpec) -> Optional[tuple[int, str]]:
    """Published depth from the checkpoint config, per the family's depth keys."""
    for scope in _config_scopes(hf_config):
        for key in spec.depth_keys:
            depth = _int_or_none(scope.get(key))
            if depth:
                return depth, f"config key {key!r}"
    if spec.declared_depth:
        return spec.declared_depth, "registration"
    return None


def _weights_depth(keys: set[str], spec: DraftModelSpec) -> Optional[int]:
    """Number of draft layers the weights actually publish."""
    if spec.weight_map.layer_key_pattern is None:
        return None
    pattern = re.compile(spec.weight_map.layer_key_pattern)
    depths = {int(m.group("depth")) for key in keys if (m := pattern.match(key))}
    return len(depths) if depths else None


def _matches_family_keys(keys: set[str], spec: DraftModelSpec) -> bool:
    """Whether the checkpoint publishes this family's characteristic keys."""
    if spec.weight_map is None:
        return False
    pattern = re.compile(spec.weight_map.family_keys)
    return any(pattern.match(key) for key in keys)


def _families_matching_layout(keys: set[str]) -> list[str]:
    """Families whose draft layout the checkpoint's tensors resemble."""
    return [
        spec.family
        for spec in DRAFT_MODEL_SPECS.values()
        if _matches_family_keys(keys, spec)
    ]


def _layer_kinds(hf_config: dict, spec: DraftModelSpec, depth: int) -> tuple:
    """Layer kinds of the draft block, from the checkpoint or the description."""
    if spec.layer_kinds:
        return spec.layer_kinds
    if spec.layer_config_key:
        for scope in _config_scopes(hf_config):
            declared = scope.get(spec.layer_config_key)
            if not isinstance(declared, list) or not declared:
                continue
            kinds = []
            for name in declared[:depth]:
                try:
                    kinds.append(DraftLayerKind(name))
                except ValueError as error:
                    raise UnsupportedDraftError(
                        f"{spec.family!r} declares draft layer type {name!r}, "
                        "which is not a known draft layer kind; describe it "
                        "before integrating the family"
                    ) from error
            return tuple(kinds)
    return ()


def _check_rejected_flags(
    hf_config: dict,
    spec: DraftModelSpec,
    error: type[Exception] = UnsupportedDraftError,
) -> None:
    for key, value, message in spec.rejected_config_flags:
        for scope in _config_scopes(hf_config):
            if scope.get(key) == value:
                raise error(message)


def _check_description(spec: DraftModelSpec) -> None:
    """Reject semantics this build never executes, naming what is missing.

    These checks depend on the description alone, so every path that uses a
    description applies them — a checkpoint that embeds its head and a
    standalone draft directory alike.
    """
    if not spec.is_available:
        raise UnsupportedDraftError(
            f"{spec.family!r} is a recognised MTP draft, but this build cannot "
            "construct it yet: missing "
            + "; ".join(spec.unimplemented)
            + '. Add the draft model (MODELS.md, "Adding a new MTP draft '
            'model") and clear `unimplemented` on its description.'
        )
    if not spec.chain_causal:
        raise UnsupportedDraftError(
            f"{spec.family!r} predicts several tokens with parallel heads; the "
            "draft path here is a serial chain, one draft layer per drafted"
            f" token (MODELS.md, {CRITERIA_SECTION})"
        )
    if spec.hidden_streams != 1:
        raise UnsupportedDraftError(
            f"{spec.family!r} carries {spec.hidden_streams} hidden streams; the "
            "draft path here consumes a single target hidden state"
        )
    if spec.fusion != FusionMode.CONCAT_PROJECTION:
        raise UnsupportedDraftError(
            f"{spec.family!r} fuses its inputs with {spec.fusion.value!r}; the "
            "draft blocks in this build concatenate the two streams and project"
        )
    if spec.concat_order == ConcatOrder.UNSPECIFIED:
        raise UnsupportedDraftError(
            f"{spec.family!r} does not publish the order in which it "
            "concatenates the embedding and hidden streams; read the family's "
            "modeling code and record it before integrating the family"
        )
    if spec.concat_order != ConcatOrder.EMBEDDING_FIRST and (
        spec.layer_source != LayerSource.FAMILY_SPECIFIC
    ):
        raise UnsupportedDraftError(
            f"{spec.family!r} concatenates {spec.concat_order.value!r}; the "
            "shared draft block concatenates the embedding first, so only a "
            "family with a draft block of its own can run this order"
        )
    if spec.embedding_sharing == EmbeddingSharing.PER_DEPTH:
        raise UnsupportedDraftError(
            f"{spec.family!r} stores one embedding per draft depth; the draft "
            "loader here shares a single embedding table with the target"
        )
    if spec.runtime_depth is not None or spec.runtime_depth_keys:
        raise UnsupportedDraftError(
            f"{spec.family!r} rolls out fewer depths than it publishes "
            f"(runtime depth {spec.runtime_depth!r} via "
            f"{list(spec.runtime_depth_keys)}); this build rolls out the"
            f" published depth (MODELS.md, {CRITERIA_SECTION})"
        )
    if spec.shared_layer_depths:
        raise UnsupportedDraftError(
            f"{spec.family!r} lets depths {list(spec.shared_layer_depths)} "
            "reuse the target's layer instead of their own; this build gives"
            " every draft depth its own layer"
            f" (MODELS.md, {CRITERIA_SECTION})"
        )


def _check_resolved(spec: DraftModelSpec, layer_kinds: tuple) -> None:
    """Reject a resolved checkpoint whose block this build cannot compose."""
    _check_description(spec)
    if layer_kinds and DraftLayerKind.FULL_ATTENTION not in layer_kinds:
        raise UnsupportedDraftError(
            f"{spec.family!r} publishes {[kind.value for kind in layer_kinds]} "
            "draft layers; this build composes full-attention draft layers only "
            "(MODELS.md, criterion C4)"
        )
    if spec.embedded and spec.weight_map is None:
        raise UnsupportedDraftError(
            f"{spec.family!r} embeds its draft weights but its description has "
            "no key mapping"
        )


def _materialize_fixture(
    model_path: str, spec: DraftModelSpec, depth: int, layer_kinds: tuple
) -> str:
    """Write a standalone draft config next to the checkpoint's own shards.

    A checkpoint that embeds its draft weights only publishes them inside the
    target config, so the description supplies the standalone shape. The
    fixture symlinks the checkpoint shards, so it must outlive the draft
    engine; the temporary directory is intentionally not cleaned up.
    """
    hf_config = _read_config(model_path)
    fixture = tempfile.mkdtemp(prefix="infinilm_draft_fixture_")
    text_config = dict(hf_config.get("text_config", hf_config))
    hf_config = dict(hf_config)
    hf_config["model_type"] = spec.draft_model_type
    text_config["num_hidden_layers"] = depth
    if spec.layer_config_key:
        kinds = layer_kinds or (DraftLayerKind.FULL_ATTENTION,) * depth
        if len(kinds) == 1:
            kinds = kinds * depth
        text_config[spec.layer_config_key] = [kind.value for kind in kinds[:depth]]
    hf_config["text_config"] = text_config
    with open(os.path.join(fixture, "config.json"), "w") as f:
        json.dump(hf_config, f)
    for name in os.listdir(model_path):
        if name.endswith(".safetensors") or name == "model.safetensors.index.json":
            os.symlink(os.path.join(model_path, name), os.path.join(fixture, name))
    return fixture


def _resolve(
    model_path: str, hf_config: dict, spec: DraftModelSpec, keys: set[str]
) -> DraftCheckpoint:
    _check_rejected_flags(hf_config, spec)
    if not _matches_family_keys(keys, spec):
        raise UnsupportedDraftError(
            f"{model_path} does not publish {spec.family!r} draft weights: no "
            f"tensor matches the family's draft keys "
            f"{spec.weight_map.family_keys!r} (MODELS.md, criterion C1)"
        )
    matched = [key for key in keys if re.match(spec.weight_map.key_pattern, key)]
    if not matched:
        raise UnsupportedDraftError(
            f"{model_path} publishes {spec.family!r} draft keys but none of "
            f"them can be renamed into the draft model: no tensor matches "
            f"{spec.weight_map.key_pattern!r}"
        )
    weights_depth = _weights_depth(keys, spec)
    config_depth = _config_depth(hf_config, spec)
    if config_depth is None and weights_depth is None:
        raise UnsupportedDraftError(
            f"{model_path} has {spec.family!r} draft weights but its depth is "
            "not determinable: the config carries none of "
            f"{list(spec.depth_keys)} and the registration declares no depth "
            "(MODELS.md, criterion C2)"
        )
    if weights_depth is not None:
        if config_depth is not None and config_depth[0] != weights_depth:
            logger.warning(
                "%s: %s says %d draft layer(s) but the weights publish %d; "
                "using the weights",
                model_path,
                config_depth[1],
                config_depth[0],
                weights_depth,
            )
        depth, depth_source = weights_depth, "weights"
    else:
        depth, depth_source = config_depth[0], config_depth[1]
    if depth < 1:
        raise UnsupportedDraftError(
            f"{model_path} declares {depth} {spec.family!r} draft layers; there "
            "is nothing to draft with"
        )
    layer_kinds = _layer_kinds(hf_config, spec, depth)
    _check_resolved(spec, layer_kinds)
    engine_path = model_path
    if spec.embedded and spec.draft_model_type:
        engine_path = _materialize_fixture(model_path, spec, depth, layer_kinds)
    return DraftCheckpoint(
        spec=spec,
        engine_path=engine_path,
        published_depth=depth,
        depth_source=depth_source,
        layer_kinds=layer_kinds,
    )


def resolve_embedded_draft(model_path: str) -> Optional[DraftCheckpoint]:
    """Resolve a checkpoint that embeds its draft weights, else None.

    A checkpoint belongs to a family when its own ``model_type`` is declared by
    that description *and* it publishes the family's characteristic draft keys;
    a matching tensor layout alone never decides which family a checkpoint is,
    so a family without a description cannot be mistaken for a described one.
    """
    hf_config = _read_config(model_path)
    if hf_config is None:
        return None
    keys = _published_keys(model_path)
    if keys is None:
        return None
    model_type = hf_config.get("model_type", "")
    for spec in DRAFT_MODEL_SPECS.values():
        if not spec.embedded or spec.weight_map is None:
            continue
        if spec.draft_model_type == model_type:
            # The checkpoint is already the standalone draft itself.
            return None
        if model_type not in spec.target_model_types:
            continue
        if not _matches_family_keys(keys, spec):
            continue
        return _resolve(model_path, hf_config, spec, keys)
    return None


def explain_missing_draft(model_path: str, model_type: str) -> str:
    """Explain why a draft engine's model type is not a supported draft.

    The three reasons are kept apart on purpose: a declared family whose
    checkpoint publishes no draft tensors (C1 does not hold), a checkpoint that
    publishes a draft layout no description covers (the family is not
    described), and a model type with nothing draft-like in it at all.
    """
    families = [
        spec.family
        for spec in DRAFT_MODEL_SPECS.values()
        if model_type in spec.target_model_types
    ]
    if families:
        return (
            f"{model_path} is a {families[0]!r} checkpoint but does not publish "
            "draft weights for it (MODELS.md, criterion C1: the checkpoint must "
            "carry the family's draft tensors)"
        )
    keys = _published_keys(model_path)
    resembling = _families_matching_layout(keys) if keys else []
    if resembling:
        return (
            f"the checkpoint publishes draft tensors resembling the layout of "
            f"{resembling}, but model type {model_type!r} has no draft "
            "description, so its draft head cannot be attributed to a family. "
            'Register a DraftModelSpec for it (MODELS.md, "Adding a new MTP '
            'draft model")'
        )
    known = sorted(
        {
            target
            for spec in DRAFT_MODEL_SPECS.values()
            for target in spec.target_model_types
        }
    )
    return (
        f"model type {model_type!r} has no draft description; registered draft "
        f"families cover the target model types {known}. Add one by registering "
        'a DraftModelSpec (MODELS.md, "Adding a new MTP draft model"); a model '
        "type outside that list is usually a checkpoint that publishes no draft "
        "weights (criterion C1)"
    )


def resolve_draft(model_path: str) -> Optional[DraftCheckpoint]:
    """Resolve the draft a directory provides, before any engine is built.

    Returns the description of a checkpoint that embeds its draft weights, None
    when the directory is a standalone draft checkpoint (the engine's own model
    type decides then), and raises when the directory carries a draft head this
    registry cannot place: either a declared family whose checkpoint publishes
    no draft tensors, or draft tensors whose family has no description. Both
    failures name what is missing instead of surfacing as a model-construction
    error or as a wrong attribution.
    """
    checkpoint = resolve_embedded_draft(model_path)
    if checkpoint is not None:
        return checkpoint
    hf_config = _read_config(model_path)
    if hf_config is None:
        return None
    model_type = hf_config.get("model_type", "")
    standalone = get_draft_model_spec(model_type)
    if standalone is not None and standalone.draft_model_type == model_type:
        # The directory is the standalone draft itself, or a fixture this
        # module materialised from an embedding checkpoint: nothing to build
        # on top of it, and the engine's own model type decides. A family
        # whose block is missing still fails, naming that block.
        _check_description(standalone)
        return None
    keys = _published_keys(model_path)
    if keys is None:
        return None
    belongs_to_family = any(
        model_type in spec.target_model_types for spec in DRAFT_MODEL_SPECS.values()
    )
    if belongs_to_family or _families_matching_layout(keys):
        raise UnsupportedDraftError(explain_missing_draft(model_path, model_type))
    return None


def get_draft_weight_remapper(model_type: str) -> Optional[Callable]:
    """Load-time weight mapper for a draft model type, or None.

    The mapper is derived from the family's description, so a new family needs
    no new entry in the loader's remapper table.
    """
    spec = get_draft_model_spec(model_type)
    if spec is None or spec.weight_map is None:
        return None
    if spec.family not in _DERIVED_REMAPPERS:

        def remap(state_dict: dict, config: dict) -> dict:
            return _remap_draft_weights(spec, state_dict, config)

        remap.__name__ = f"_remap_{spec.family}_draft"
        remap.__doc__ = f"Apply {spec.family} draft load-time weight fixes."
        _DERIVED_REMAPPERS[spec.family] = remap
    return _DERIVED_REMAPPERS[spec.family]


def _embedded_family_for_target(model_type: str) -> Optional[DraftModelSpec]:
    """Description that answers for a target embedding this family's head."""
    for spec in DRAFT_MODEL_SPECS.values():
        if (
            spec.embedded
            and spec.weight_map is not None
            and model_type in spec.target_model_types
        ):
            return spec
    return None


def _draft_namespace_pattern(spec: DraftModelSpec) -> "re.Pattern[str]":
    """Key pattern of the tensors that belong to this family's draft head."""
    return re.compile(spec.weight_map.family_keys)


def _draft_layer_prefixes(spec: DraftModelSpec, state_dict: dict) -> tuple[str, ...]:
    """Layer prefixes this family's own keys locate in the published weights.

    A family whose draft block reuses the target's decoder-layer layout
    publishes its draft layers inside the ``model.layers.<N>.`` namespace every
    checkpoint has, so the key feature alone does not name every tensor of the
    block. Those layers are located through the family's own
    ``layer_key_pattern`` instead: wherever it matches, the layer index it
    exposes marks a prefix, and every tensor under that prefix belongs to the
    same draft layer.

    The prefix carries the separator that follows the layer index, so a prefix
    for layer 6 never swallows a tensor of layer 61. A checkpoint that
    publishes none of these keys locates nothing, which leaves the key feature
    as the whole rule and keeps the removal from growing on other checkpoints.
    A ``layer_key_pattern`` that does not name the index in a ``depth`` group —
    a description that breaks the contract the field is documented with — is
    reported by family name instead of surfacing as a group lookup error.
    """
    pattern_text = spec.weight_map.layer_key_pattern
    if pattern_text is None:
        return ()
    pattern = re.compile(pattern_text)
    if "depth" not in pattern.groupindex:
        # A description contract rather than a checkpoint property: a pattern
        # that cannot name the layer index cannot locate a draft layer, and
        # saying so here keeps the target side from failing with a bare lookup
        # error inside a weight load.
        raise ValueError(
            f"the {spec.family} description must expose its draft layer index in "
            f"a group named 'depth' (layer_key_pattern={pattern_text!r})"
        )
    prefixes: list[str] = []
    for key in state_dict:
        match = pattern.match(key)
        if match is None:
            continue
        prefix = key[: match.end("depth")] + "."
        if prefix not in prefixes:
            prefixes.append(prefix)
    return tuple(prefixes)


def _drop_draft_namespace(spec: DraftModelSpec, state_dict: dict) -> dict:
    """Keep every tensor except the draft tensors this family publishes."""
    pattern = _draft_namespace_pattern(spec)
    prefixes = _draft_layer_prefixes(spec, state_dict)
    return {
        key: tensor
        for key, tensor in state_dict.items()
        if not pattern.match(key) and not key.startswith(prefixes)
    }


def get_embedded_draft_target_remapper(model_type: str) -> Optional[Callable]:
    """Load-time weight mapper for the *target* of an embedded draft family.

    A checkpoint that embeds its draft head is loaded as a whole by the target
    engine, and the embedded tensors are unknown keys to the target module tree.
    This mapper removes exactly the draft tensors the family publishes — the
    keys its own key feature recognises, plus the draft layers that feature
    locates in the checkpoint's weights — and hands every other tensor back
    unchanged, so a described family needs no entry of its own in the loader's
    remapper table.

    Only a description of a family whose head is embedded in the target
    checkpoint answers here, and the removal is driven by the family's
    characteristic keys rather than by a layout every checkpoint shares, so a
    tensor outside the family's draft tensors is never removed. The mapper is
    derived from the family description, so re-registering that description
    retires the mapper with it.
    """
    spec = _embedded_family_for_target(model_type)
    if spec is None:
        return None
    if spec.family not in _DERIVED_TARGET_REMAPPERS:

        def remap(state_dict: dict, config: dict = None) -> dict:
            return _drop_draft_namespace(spec, state_dict)

        remap.__name__ = f"_remap_{spec.family}_target"
        remap.__doc__ = (
            f"Drop the {spec.family} draft tensors an embedded target checkpoint "
            "publishes."
        )
        _DERIVED_TARGET_REMAPPERS[spec.family] = remap
    return _DERIVED_TARGET_REMAPPERS[spec.family]


def _remap_draft_weights(spec: DraftModelSpec, state_dict: dict, config: dict) -> dict:
    weight_map = spec.weight_map
    # A description that cannot express what the checkpoint declares must fail
    # loudly here as well, so the loader never silently drops such weights.
    _check_rejected_flags(config, spec, error=NotImplementedError)
    text_config = config.get("text_config", config)
    depth_index = None
    if weight_map.depth_index_key:
        depth_index = text_config.get(weight_map.depth_index_key)
    pattern = re.compile(weight_map.key_pattern)
    if depth_index is not None and "depth" not in pattern.groupindex:
        # A description contract rather than a checkpoint property, checked the
        # same way the layer-locating pattern is: a description that anchors a
        # block on a published layer index it never captured would otherwise
        # fail the weight load with a bare group lookup error.
        raise ValueError(
            f"the {spec.family} description anchors its draft block on"
            f" {weight_map.depth_index_key!r} but does not expose the published"
            f" layer index in a group named 'depth' (key_pattern="
            f"{weight_map.key_pattern!r})"
        )
    remapped: dict = {}
    for key, tensor in state_dict.items():
        match = pattern.match(key)
        if match is None:
            continue
        groups = match.groupdict()
        if depth_index is not None and int(groups["depth"]) != int(depth_index):
            continue
        new_key = weight_map.canonical_prefix + groups["rest"]
        for old, new in weight_map.renames:
            new_key = new_key.replace(old, new)
        if new_key in weight_map.zero_centered_keys or new_key.endswith(
            weight_map.zero_centered_norms
        ):
            # Zero-centered RMSNorm scales are stored around zero while
            # InfiniCore RMSNorm multiplies directly; shift to the effective
            # weight.
            tensor = tensor + torch.ones_like(tensor)
        remapped[new_key] = tensor

    embedding = _first_present(state_dict, weight_map.embedding_keys)
    if config.get("tie_word_embeddings", text_config.get("tie_word_embeddings", False)):
        # Tied checkpoints store a single matrix for the embedding and the head.
        head = embedding
    else:
        # Untied checkpoints carry the target's own output head; the draft
        # shares it with the target.
        head = _first_present(state_dict, weight_map.head_keys)
    if embedding is not None:
        remapped.setdefault("model.embed_tokens.weight", embedding)
    if head is not None:
        remapped.setdefault("lm_head.weight", head)
    return remapped


def _first_present(state_dict: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in state_dict:
            return state_dict[key]
    return None


def draft_position_ids(spec: DraftModelSpec, positions: list[int]):
    """Position ids a family's draft engine expects for a step's batch."""
    import infinicore

    if spec.position_ids == PositionIdLayout.MROPE_TEXT:
        # Text-only steps repeat the same position on every mrope axis.
        return infinicore.from_list([list(positions)] * 3, dtype=infinicore.int64)
    return infinicore.from_list([[pos] for pos in positions], dtype=infinicore.int64)


# --------------------------------------------------------------------------- #
# Family descriptions
#
# The first group runs in this build. The second group is described but its
# draft block is not implemented here yet: `unimplemented` (and the
# `is_available` property it drives) is what separates the two, and using one of
# them fails at construction with the missing piece named.
# --------------------------------------------------------------------------- #

# --- Available draft blocks ------------------------------------------------- #

QWEN35_MTP = register_draft_model_spec(
    DraftModelSpec(
        family="qwen3_5_mtp",
        draft_model_type="qwen3_5_mtp",
        target_model_types=("qwen3_5",),
        embedded=True,
        depth_keys=("mtp_num_hidden_layers",),
        layer_kinds=(DraftLayerKind.FULL_ATTENTION,),
        embedding_at_position_zero=EmbeddingAtPositionZero.PREVIOUS_TOKEN,
        recycle_hidden=RecycleHidden.POST_FINAL_NORM,
        embedding_sharing=EmbeddingSharing.SHARED_WITH_TARGET,
        layer_source=LayerSource.REUSE_TARGET,
        position_ids=PositionIdLayout.MROPE_TEXT,
        weight_map=DraftWeightMap(
            family_keys=r"^mtp\.(?:fc|norm|pre_fc_norm_embedding|pre_fc_norm_hidden|layers)\.",
            key_pattern=r"^mtp\.(?P<rest>.+)$",
            layer_key_pattern=r"^mtp\.layers\.(?P<depth>\d+)\.",
            zero_centered_keys=("model.norm.weight",),
            zero_centered_norms=(
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "pre_fc_norm_embedding.weight",
                "pre_fc_norm_hidden.weight",
            ),
            embedding_keys=(
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ),
            head_keys=("lm_head.weight",),
        ),
        rejected_config_flags=(
            (
                "mtp_use_dedicated_embeddings",
                True,
                "Qwen3.5 MTP checkpoints with dedicated draft embeddings are "
                "not supported",
            ),
        ),
    )
)

MINICPM_EAGLE = register_draft_model_spec(
    DraftModelSpec(
        family="minicpm_eagle",
        draft_model_type="minicpm_eagle",
        embedded=False,
        layer_config_key=None,
        layer_kinds=(DraftLayerKind.FULL_ATTENTION,),
        embedding_at_position_zero=EmbeddingAtPositionZero.UNSPECIFIED,
        recycle_hidden=RecycleHidden.PRE_FINAL_NORM,
        embedding_sharing=EmbeddingSharing.DRAFT_CHECKPOINT,
        layer_source=LayerSource.FAMILY_SPECIFIC,
        weight_map=None,
    )
)

# --- Described, draft block not implemented in this build ------------------- #

# Qwen3-Next and Qwen3.5-35B-A3B publish the same mtp.* layout as Qwen3.5 dense
# but with a MoE MLP inside the draft block; Qwen3-Next's config carries no MTP
# depth key at all, so the registration declares one depth.
QWEN_MOE_MTP = register_draft_model_spec(
    DraftModelSpec(
        family="qwen_moe_mtp",
        draft_model_type="qwen_moe_mtp",
        target_model_types=("qwen3_next", "qwen3_5_moe"),
        embedded=True,
        depth_keys=(),
        declared_depth=1,
        layer_kinds=(DraftLayerKind.FULL_ATTENTION, DraftLayerKind.MOE),
        recycle_hidden=RecycleHidden.POST_FINAL_NORM,
        embedding_sharing=EmbeddingSharing.SHARED_WITH_TARGET,
        layer_source=LayerSource.FAMILY_SPECIFIC,
        weight_map=DraftWeightMap(
            family_keys=r"^mtp\.(?:fc|norm|pre_fc_norm_embedding|pre_fc_norm_hidden|layers)\.",
            key_pattern=r"^mtp\.(?P<rest>.+)$",
            layer_key_pattern=r"^mtp\.layers\.(?P<depth>\d+)\.",
            zero_centered_keys=("model.norm.weight",),
            zero_centered_norms=(
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "pre_fc_norm_embedding.weight",
                "pre_fc_norm_hidden.weight",
            ),
            embedding_keys=(
                "model.language_model.embed_tokens.weight",
                "model.embed_tokens.weight",
            ),
            head_keys=("lm_head.weight",),
        ),
        unimplemented=(
            "a draft block with a MoE MLP (512 experts plus a shared expert in "
            "the published checkpoints)",
        ),
    )
)

DEEPSEEK_V3_MTP = register_draft_model_spec(
    DraftModelSpec(
        family="deepseek_v3_mtp",
        draft_model_type="deepseek_v3_mtp",
        target_model_types=("deepseek_v3", "deepseek_v32"),
        embedded=True,
        depth_keys=("num_nextn_predict_layers",),
        layer_kinds=(DraftLayerKind.MLA_ATTENTION, DraftLayerKind.MOE),
        embedding_at_position_zero=EmbeddingAtPositionZero.UNSPECIFIED,
        recycle_hidden=RecycleHidden.PRE_FINAL_NORM,
        embedding_sharing=EmbeddingSharing.TARGET_EMBEDDING_AND_HEAD,
        layer_source=LayerSource.REUSE_TARGET,
        weight_map=DraftWeightMap(
            family_keys=(r"^model\.layers\.\d+\.(?:enorm|hnorm|eh_proj|shared_head)\."),
            key_pattern=r"^model\.layers\.(?P<depth>\d+)\.(?P<rest>.+)$",
            layer_key_pattern=(
                r"^model\.layers\.(?P<depth>\d+)\."
                r"(?:enorm|hnorm|eh_proj|shared_head)\."
            ),
            depth_index_key="num_hidden_layers",
            embedding_keys=("model.embed_tokens.weight",),
            head_keys=("lm_head.weight",),
        ),
        unimplemented=(
            "a DeepSeek MLA + MoE draft block (the target's own decoder layer, "
            "reused as the MTP block)",
            "the shared-head norm convention of this family",
            "how the per-depth embed_tokens tensor this family publishes "
            "relates to the target embedding",
        ),
    )
)

# MiMo-7B publishes its draft block inside the same checkpoint, under its own
# `model.mtp_layers.<depth>.` namespace, and fuses the two input streams in the
# reverse order of the qwen3_5_mtp layout.
MIMO_MTP = register_draft_model_spec(
    DraftModelSpec(
        family="mimo_mtp",
        draft_model_type="mimo_mtp",
        target_model_types=("mimo",),
        embedded=True,
        depth_keys=("num_nextn_predict_layers",),
        layer_kinds=(DraftLayerKind.FULL_ATTENTION,),
        # Both released rollout implementations concatenate the normalized
        # target hidden state before the normalized next-token embedding.
        concat_order=ConcatOrder.HIDDEN_FIRST,
        embedding_at_position_zero=EmbeddingAtPositionZero.ZEROED,
        embedding_sharing=EmbeddingSharing.TARGET_EMBEDDING_AND_HEAD,
        layer_source=LayerSource.FAMILY_SPECIFIC,
        weight_map=DraftWeightMap(
            family_keys=r"^model\.mtp_layers\.\d+\.",
            key_pattern=r"^model\.mtp_layers\.(?P<depth>\d+)\.(?P<rest>.+)$",
            layer_key_pattern=r"^model\.mtp_layers\.(?P<depth>\d+)\.",
            renames=(
                ("token_layernorm.", "pre_fc_norm_embedding."),
                ("hidden_layernorm.", "pre_fc_norm_hidden."),
                ("input_proj.", "fc."),
                ("final_layernorm.", "norm."),
            ),
            embedding_keys=("model.embed_tokens.weight",),
            head_keys=("lm_head.weight",),
        ),
    )
)
