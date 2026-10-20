"""Generate InfiniOps PyTorch wrappers from ATen `native_functions.yaml`.

For each op listed in `scripts/torch_ops.yaml`, this script finds the `.out`
variant in PyTorch's locally installed ATen schemas, including variants declared
through `autogen`, and emits:
  - `generated/base/<op>.h` — the InfiniOps base class
    `class <Op> : public Operator<<Op>>`, with constructors and pure-virtual
    `operator()` overloads mirroring the selected ATen schemas.
  - `generated/torch/<op>/<op>.h` and `<op>.cc` — the PyTorch backend
    `Operator<<Op>, kDev, 8>` that calls `at::<op>_out(out, ...)`.
  - `generated/torch_ops_metadata.json` — the kind (`unary` / `binary` /
    `binary_alpha`) of every successfully-generated op, consumed by the
    parametrized test suite.

Slot 8 is the reserved convention for PyTorch backends; slots 0-7 are
left for native or vendor implementations.  (The slot must also be > 0
to side-step a partial-specialization-after-instantiation conflict with
the primary template `Operator<<Op>>` instantiated at index 0.)

The generated files are not committed; CMake regenerates them at configure
time when `WITH_TORCH=ON`.
"""

import argparse
import collections
import dataclasses
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

import yaml

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
import ops_config  # noqa: E402

_REPO_ROOT = _SCRIPTS_DIR.parent
_OPS_YAML_PATH = _SCRIPTS_DIR / "torch_ops.yaml"
_BASE_DIR = _REPO_ROOT / "src" / "base"
_GENERATED_DIR = _REPO_ROOT / "generated"
_GENERATED_BASE_DIR = _GENERATED_DIR / "base"
_GENERATED_TORCH_DIR = _GENERATED_DIR / "torch"
_METADATA_PATH = _GENERATED_DIR / "torch_ops_metadata.json"

# Reserved slot for PyTorch backends. Native and vendor implementations
# claim slots 0-7; PyTorch wrappers always live at 8.
_PYTORCH_SLOT = 8

# ATen uses symbolic names for some `int`/`float` defaults (e.g.
# `reduction=Mean`). Map them to C++ identifiers usable in a call.
_ENUM_DEFAULTS = {
    "Mean": "at::Reduction::Mean",
    "Sum": "at::Reduction::Sum",
    "Contiguous": "at::MemoryFormat::Contiguous",
}

# Default PyTorch schema label used only in diagnostics when CMake does not
# provide the locally installed torch version. Codegen reads the actual schema
# from installed `torchgen/packaged/ATen/native/native_functions.yaml`.
_DEFAULT_PYTORCH_VERSION = "v2.4.0"

# Order matches the device list in existing hand-written torch backends
# (see `src/torch/add/add.cc`).
_DEVICE_TYPES = (
    "kCpu",
    "kNvidia",
    "kCambricon",
    "kAscend",
    "kMetax",
    "kMoore",
    "kIluvatar",
    "kHygon",
    "kThead",
    "kMars",
)

_C10_DEVICE_TYPES = (
    "kNvidia",
    "kCambricon",
    "kAscend",
    "kMetax",
    "kMoore",
    "kIluvatar",
    "kThead",
    "kMars",
)

# YAML scalar-type tokens → C++ types. Reference types (e.g. `const Scalar&`)
# are not used so the generated signatures match the existing hand-written
# ones, which pass by value to keep pybind11 binding generation simple.
_SCALAR_TYPE_MAP = {
    # `at::Scalar` is implicitly constructible from `double`, so we expose
    # scalars as `double` in the base class to keep it torch-independent.
    "Scalar": "double",
    "int": "int64_t",
    "bool": "bool",
    "float": "double",
    # `SymInt` / `SymInt[]` exist for `torch.compile` internals; at runtime
    # they're just `int64`/IntArrayRef.
    "SymInt": "int64_t",
    # `str` for required string params (e.g. `index_reduce.reduce`).
    # `std::string` marshals through pybind11 cleanly and converts
    # implicitly to ATen's `c10::string_view`.
    "str": "std::string",
    "ScalarType": "DataType",
}

# `Dimname` overloads (named-tensor dim) are skipped — passing them
# from Python to ATen requires a wrapper conversion through
# `at::Dimname::fromSymbol(...)` that doesn't fit the cleanly-rendered
# 1:1 arg model, and named tensors remain experimental in PyTorch.
# The int-dim overload is always emitted alongside, so we lose nothing
# user-visible.

# Optional ATen types and their typed empty optional values at the call site.
# We use typed `c10::optional<T>{}` values rather than bare `at::nullopt` so
# the compiler can disambiguate ops with multiple `_out` overloads (e.g.
# `clamp_out` accepts both `optional<Scalar>` and `optional<Tensor>` for
# `min`/`max`).
_NULLOPT_BY_TYPE = {
    "Scalar?": "c10::optional<at::Scalar>{}",
    "int?": "c10::optional<int64_t>{}",
    "bool?": "c10::optional<bool>{}",
    "float?": "c10::optional<double>{}",
    "str?": "c10::optional<c10::string_view>{}",
    "ScalarType?": "c10::optional<at::ScalarType>{}",
    "MemoryFormat?": "c10::optional<at::MemoryFormat>{}",
    "Layout?": "c10::optional<at::Layout>{}",
    "Device?": "c10::optional<at::Device>{}",
    "Generator?": "c10::optional<at::Generator>{}",
    "Tensor?": "c10::optional<at::Tensor>{}",
    "Tensor?[]": "c10::List<c10::optional<at::Tensor>>{}",
    "int[]?": "c10::optional<at::IntArrayRef>{}",
    "int[1]?": "c10::optional<at::IntArrayRef>{}",
    "int[2]?": "c10::optional<at::IntArrayRef>{}",
    "int[3]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt?": "c10::optional<int64_t>{}",
    "SymInt[]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt[1]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt[2]?": "c10::optional<at::IntArrayRef>{}",
    "SymInt[3]?": "c10::optional<at::IntArrayRef>{}",
    "float[]?": "c10::optional<at::ArrayRef<double>>{}",
}

# Optional ATen types that have a stable InfiniOps representation. These are
# exposed in generated base signatures and converted back to ATen optionals in
# the PyTorch backend. PyTorch-internal concepts without a public InfiniOps
# analogue stay hidden and are still passed as typed empty optionals.
_EXPOSED_OPTIONAL_CPP_TYPES = {
    "Scalar?": "std::optional<double>",
    "int?": "std::optional<int64_t>",
    "bool?": "std::optional<bool>",
    "float?": "std::optional<double>",
    "str?": "std::optional<std::string>",
    "ScalarType?": "std::optional<DataType>",
    "Tensor?": "std::optional<Tensor>",
    "Tensor?[]": "std::vector<std::optional<Tensor>>",
    "int[]?": "std::optional<std::vector<int64_t>>",
    "int[1]?": "std::optional<std::vector<int64_t>>",
    "int[2]?": "std::optional<std::vector<int64_t>>",
    "int[3]?": "std::optional<std::vector<int64_t>>",
    "SymInt?": "std::optional<int64_t>",
    "SymInt[]?": "std::optional<std::vector<int64_t>>",
    "SymInt[1]?": "std::optional<std::vector<int64_t>>",
    "SymInt[2]?": "std::optional<std::vector<int64_t>>",
    "SymInt[3]?": "std::optional<std::vector<int64_t>>",
    "float[]?": "std::optional<std::vector<double>>",
}
_HARDCODE_NULLOPT_TYPES = frozenset(
    set(_NULLOPT_BY_TYPE) - set(_EXPOSED_OPTIONAL_CPP_TYPES)
)


def _normalize_cpp_type(cpp_type: str) -> str:
    text = cpp_type.strip()

    if text.startswith("const "):
        text = text[len("const ") :]

    return text.rstrip("&").strip()


def _write_text_if_changed(path: pathlib.Path, content: str) -> bool:
    """Write `content` only when the file's bytes would change."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.read_text() == content:
        return False

    path.write_text(content)
    return True


def _prune_empty_dirs(root: pathlib.Path) -> None:
    if not root.exists():
        return

    for child in sorted(root.iterdir(), reverse=True):
        if child.is_dir():
            _prune_empty_dirs(child)

    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def _remove_stale_files(root: pathlib.Path, expected_files: set[pathlib.Path]) -> None:
    if not root.exists():
        return

    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and path not in expected_files:
            path.unlink()

    _prune_empty_dirs(root)


@dataclasses.dataclass
class Param:
    name: str
    aten_type: str
    default: str | None
    keyword_only: bool
    cpp_type_override: str | None = None

    @property
    def api_name(self) -> str:
        if self.name == "self":
            return "input"

        return self.name

    @property
    def is_tensor(self) -> bool:
        if self.cpp_type_override is not None:
            return _normalize_cpp_type(self.cpp_type_override) == "Tensor"

        # Real tensors only. `Tensor?` and `Tensor[]` have separate handling.
        if self.aten_type.endswith("[]"):
            return False

        return self.aten_type == "Tensor" or self.aten_type.startswith("Tensor(")

    @property
    def is_optional_tensor(self) -> bool:
        if self.cpp_type_override is not None:
            return (
                _normalize_cpp_type(self.cpp_type_override) == "std::optional<Tensor>"
            )

        return self.aten_type == "Tensor?"

    @property
    def is_tensor_list(self) -> bool:
        if self.cpp_type_override is not None:
            return _normalize_cpp_type(self.cpp_type_override) == "std::vector<Tensor>"

        return self.aten_type == "Tensor[]"

    @property
    def is_optional_tensor_list(self) -> bool:
        if self.cpp_type_override is not None:
            return (
                _normalize_cpp_type(self.cpp_type_override)
                == "std::vector<std::optional<Tensor>>"
            )

        return self.aten_type == "Tensor?[]"

    @property
    def is_mutable_tensor(self) -> bool:
        # Mutable tensors carry `!` in their alias annotation, e.g.
        # `Tensor(a!)`.

        return self.is_tensor and "!" in self.aten_type

    @property
    def is_out(self) -> bool:
        # In ATen `_out` schemas, output tensors are keyword-only mutable tensor
        # params. Some mutable tensors are real inputs (`running_mean` /
        # `running_var` in `_batch_norm_with_update`) and must stay in schema
        # order, so mutability alone is not enough.

        return self.is_mutable_tensor and self.keyword_only

    @property
    def is_hardcoded_nullopt(self) -> bool:
        """If `True`, the param is omitted from the user-facing API and
        passed as `at::nullopt` to ATen."""

        return self.aten_type in _HARDCODE_NULLOPT_TYPES

    @property
    def is_hidden(self) -> bool:
        """True if the param is omitted from the user-facing API.

        Default-valued non-optional params (\\`bool\\`, \\`int\\`, \\`float\\`,
        \\`str\\`, \\`int[N]\\`, …) used to be hidden as a convenience, but
        reviewers consistently flagged the resulting omissions —
        \\`bool upper/transpose/unitriangular\\` on \\`triangular_solve\\`,
        \\`int diagonal\\` on \\`triu\\`, \\`str ord\\` on \\`linalg_matrix_norm\\`,
        \\`int n\\` on the special chebyshev family, etc. — as missing
        semantic controls.  They are now exposed and forwarded to ATen.

        Optional ATen types with a stable InfiniOps representation
        (\\`Tensor?\\`, \\`Scalar?\\`, \\`int?\\`, …) are exposed as
        \\`std::optional\\`. ATen-internal types like
        \\`Generator?\\`/\\`Layout?\\` still have no InfiniOps analogue, so
        they remain hidden.
        """

        return self.is_hardcoded_nullopt

    def hidden_value(self) -> str:
        """C++ literal substituted for a hidden param in the ATen call."""

        if self.is_hardcoded_nullopt:
            return _NULLOPT_BY_TYPE[self.aten_type]

        if self.default == "True":
            return "true"

        if self.default == "False":
            return "false"

        if self.aten_type.startswith(("int[", "SymInt[")) and self.default is not None:
            # `int[N]=[a, b, c]` → `{a, b, c}`; `int[N]=0` (scalar default
            # for list type) → `{0, 0, ...}` replicated to size N.
            if self.default.startswith("["):
                return "{" + self.default[1:-1] + "}"

            size_match = re.search(r"\[(\d+)\]", self.aten_type)
            n = int(size_match.group(1)) if size_match else 1

            return "{" + ", ".join([self.default] * n) + "}"

        if self.aten_type == "str" and self.default is not None:
            # YAML uses single-quoted strings (e.g. `'none'`); C++ char
            # literals also use single quotes, so swap to doubles.

            return '"' + self.default.strip("'\"") + '"'

        if self.aten_type in {"int", "float", "SymInt"} and self.default is not None:
            # Translate known ATen enum defaults to their C++ identifiers.

            return _ENUM_DEFAULTS.get(self.default, self.default)

        raise AssertionError(
            f"param {self.name!r} of type {self.aten_type!r} with default "
            f"{self.default!r} is not hidden"
        )

    @property
    def cpp_type(self) -> str:
        if self.cpp_type_override is not None:
            return self.cpp_type_override

        if self.is_tensor:
            return "Tensor"

        if self.is_tensor_list:
            return "std::vector<Tensor>"

        if self.aten_type in _EXPOSED_OPTIONAL_CPP_TYPES:
            return _EXPOSED_OPTIONAL_CPP_TYPES[self.aten_type]

        if self.is_hidden:
            # Not exposed — the ATen call substitutes a hardcoded value
            # so the `cpp_type` is irrelevant.

            return "void"

        bare = self.aten_type.rstrip("?")
        # Required `int[N]` / `SymInt[N]` (no default) — pybind11 accepts
        # a Python list of ints into `std::vector<int64_t>`, which ATen
        # promotes to `IntArrayRef` implicitly.
        if bare.startswith(("int[", "SymInt[")) or bare in {"int[]", "SymInt[]"}:
            return "std::vector<int64_t>"

        try:
            return _SCALAR_TYPE_MAP[bare]
        except KeyError as exc:
            raise NotImplementedError(
                f"unsupported ATen type {self.aten_type!r} for param {self.name!r}"
            ) from exc


@dataclasses.dataclass
class Op:
    aten_name: str
    overload: str
    params: list[Param]
    signature_params: list[Param] | None = None
    param_bindings: list[Param | None] | None = None
    explicit_inplace_out: Param | None = None

    @property
    def pascal_name(self) -> str:
        return _snake_to_pascal(self.infini_name)

    @property
    def infini_name(self) -> str:
        """InfiniOps public op name.

        ATen disambiguates `_out` overloads with suffixes like `Tensor_Tensor_out`,
        `out_x`, `forward_output`, `grad_input`, but reviewers consistently
        flag those suffixes as bad public-API naming when they leak into
        InfiniOps class names.  Different ATen overloads of the same base op
        become overloaded `operator()` methods on a single class instead.  When
        two overloads collapse to the same visible C++ signature after hidden
        defaults, `_dedupe_visible_overloads` keeps only one.

        ATen-internal leading underscores and in-place trailing underscores are
        also normalized so generated class names do not collide with existing
        public ops (`_softmax` → `AtenSoftmax`, `add_` → `AddInplace`).
        """

        return _public_op_name(self.aten_name)

    @property
    def is_inplace(self) -> bool:
        """True for ATen's single-underscore in-place variants."""

        return _is_inplace_aten_name(self.aten_name)

    @property
    def tensor_params(self) -> list[Param]:
        return [p for p in self.params if p.is_tensor]

    @property
    def out_params(self) -> list[Param]:
        """Mutable tensor outputs.  Most ops have one (`Tensor(a!) out`);
        multi-output ops like `frexp` or `sort` have several
        (`Tensor(a!) values`, `Tensor(b!) indices`)."""

        if self.explicit_inplace_out is not None:
            return [self.explicit_inplace_out]

        if self.is_inplace:
            return [self.params[0]]

        return [p for p in self.params if p.is_out]

    @property
    def out_param(self) -> Param:
        """Single-output convenience.  Asserts there's exactly one."""
        outs = self.out_params
        assert len(outs) == 1, f"op {self.aten_name!r} has {len(outs)} out tensors"

        return outs[0]

    @property
    def visible_params(self) -> list[Param]:
        """Params the wrapper exposes to the user; hidden ones (hardcoded
        optional nullopt, default-`False`/`True` bools) are filtered."""

        if self.signature_params is not None:
            return self.signature_params

        return [p for p in self.params if not p.is_hidden]

    def api_param_for(self, schema_index: int) -> Param | None:
        """Return the public signature param bound to a schema param."""

        if self.param_bindings is not None:
            return self.param_bindings[schema_index]

        param = self.params[schema_index]

        if param.is_hidden:
            return None

        return param

    @property
    def is_testable(self) -> bool:
        """Cheap structural check: at least one out tensor, and the first
        constructor parameter is a tensor.  The latter is needed because
        `Operator::Make(Tensor tensor, Args... args)` dispatches on
        `tensor.device()`, so an op like `pow.Scalar_out(Scalar self,
        Tensor exponent, *, Tensor(a!) out)` cannot be wired up without
        a separate dispatch path.  Generators like `arange` / `linspace`
        also fall under this rule (no input tensors at all)."""

        if not self.out_params:
            return False

        if self.is_inplace:
            return self.params[0].is_mutable_tensor

        # `params` includes out tensors at the end; check the first
        # non-out param. If there are no non-out params (`empty.out`,
        # `arange.out`), this op also fails the dispatch precondition.
        non_out = [p for p in self.params if not p.is_out]

        if not non_out:
            return False

        return non_out[0].is_tensor or non_out[0].is_tensor_list


_FUNC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)"
    r"(?:\.(?P<overload>\w+))?"
    r"\((?P<args>.*)\)\s*->\s*.+$"
)

_ARG_RE = re.compile(
    r"^(?P<type>\S+(?:\([^)]*\))?\??)"  # type with optional alias and `?`
    r"\s+(?P<name>\w+)"
    r"(?:\s*=\s*(?P<default>.+))?$"
)


def _parse_func(func_str: str) -> Op:
    m = _FUNC_RE.match(func_str)

    if not m:
        raise ValueError(f"could not parse func: {func_str!r}")

    return Op(
        aten_name=m.group("name"),
        overload=m.group("overload") or "",
        params=_parse_args(m.group("args")),
    )


def _parse_args(args_str: str) -> list[Param]:
    params: list[Param] = []
    keyword_only = False

    for token in _split_args(args_str):
        if token == "*":
            keyword_only = True
            continue

        params.append(_parse_one_arg(token, keyword_only))

    return params


def _split_args(args_str: str) -> list[str]:
    """Split on top-level commas, respecting `(...)` and `[...]`."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in args_str:
        if ch in "([":
            depth += 1
            current.append(ch)
        elif ch in ")]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(current).strip()

            if piece:
                parts.append(piece)

            current = []
        else:
            current.append(ch)

    tail = "".join(current).strip()

    if tail:
        parts.append(tail)

    return parts


def _parse_one_arg(token: str, keyword_only: bool) -> Param:
    m = _ARG_RE.match(token)

    if not m:
        raise ValueError(f"could not parse arg: {token!r}")

    name = m.group("name")
    return Param(
        name=name,
        aten_type=m.group("type"),
        default=m.group("default"),
        keyword_only=keyword_only,
    )


_BASE_OPERATOR_RE = re.compile(
    r"\bvirtual\s+void\s+operator\(\)\s*\((?P<args>.*?)\)\s*const"
    r"(?:\s*=\s*0|\s*\{)",
    re.S,
)


def _split_cpp_params(args_str: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []

    for ch in args_str:
        if ch in "([<":
            depth += 1
            current.append(ch)
        elif ch in ")]>":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(current).strip()

            if piece:
                parts.append(piece)

            current = []
        else:
            current.append(ch)

    tail = "".join(current).strip()

    if tail:
        parts.append(tail)

    return parts


def _parse_base_signature_param(token: str) -> tuple[str, str]:
    token = token.split("=", 1)[0].strip()
    token = re.sub(r"\s+", " ", token)
    match = re.match(r"(?P<type>.+?)\s+(?P<name>\w+)$", token)

    if not match:
        raise ValueError(f"could not parse base operator param: {token!r}")

    return _normalize_cpp_type(match.group("type")), match.group("name")


def _parse_base_operator_signatures(
    base_path: pathlib.Path,
) -> list[list[tuple[str, str]]]:
    source = base_path.read_text()
    signatures = []

    for match in _BASE_OPERATOR_RE.finditer(source):
        args = match.group("args").strip()

        if not args:
            signatures.append([])
            continue

        signatures.append(
            [_parse_base_signature_param(token) for token in _split_cpp_params(args)]
        )

    return signatures


def _cpp_types_compatible(schema_cpp_type: str, base_cpp_type: str) -> bool:
    schema = _normalize_cpp_type(schema_cpp_type)
    base = _normalize_cpp_type(base_cpp_type)

    if schema == base:
        return True

    compatible_pairs = {
        ("double", "float"),
        ("int64_t", "int"),
        ("std::optional<double>", "std::optional<float>"),
        ("std::optional<int64_t>", "std::optional<int>"),
    }

    return (schema, base) in compatible_pairs


def _cpp_param_compatible(
    schema_param: Param, base_cpp_type: str, base_name: str
) -> bool:
    if schema_param.api_name != base_name:
        return False

    return _cpp_types_compatible(schema_param.cpp_type, base_cpp_type)


def _bind_base_signature(op: Op, signature: list[tuple[str, str]]) -> Op | None:
    bindings: list[Param | None] = [None] * len(op.params)
    signature_params = []
    explicit_inplace_out = None

    if (
        op.is_inplace
        and signature
        and signature[-1][1] == "out"
        and _cpp_types_compatible("Tensor", signature[-1][0])
    ):
        out_cpp_type, out_name = signature[-1]
        explicit_inplace_out = Param(
            name=out_name,
            aten_type="Tensor(a!)",
            default=None,
            keyword_only=True,
            cpp_type_override=out_cpp_type,
        )
        signature = signature[:-1]

    for base_cpp_type, base_name in signature:
        matched_index = None

        for index, schema_param in enumerate(op.params):
            if bindings[index] is not None:
                continue

            if _cpp_param_compatible(schema_param, base_cpp_type, base_name):
                matched_index = index
                break

        if matched_index is None:
            return None

        schema_param = op.params[matched_index]
        api_param = dataclasses.replace(
            schema_param, name=base_name, cpp_type_override=base_cpp_type
        )
        bindings[matched_index] = api_param
        signature_params.append(api_param)

    if any(
        binding is None and not _is_omittable_param(param)
        for param, binding in zip(op.params, bindings)
    ):
        return None

    if explicit_inplace_out is not None:
        signature_params.append(explicit_inplace_out)

    return dataclasses.replace(
        op,
        signature_params=signature_params,
        param_bindings=bindings,
        explicit_inplace_out=explicit_inplace_out,
    )


def _bind_existing_base_overloads(
    name: str, ops: list[Op]
) -> tuple[list[Op], list[str]]:
    base_path = _base_path(name)
    signatures = _parse_base_operator_signatures(base_path)
    bound_ops = []
    warnings = []

    for signature in signatures:
        matches = [
            bound
            for op in ops
            if (bound := _bind_base_signature(op, signature)) is not None
        ]

        if not matches:
            rendered = ", ".join(f"{cpp_type} {param}" for cpp_type, param in signature)
            warnings.append(
                f"base overload `operator()({rendered})` does not match any "
                "usable ATen schema"
            )
            continue

        matches.sort(
            key=lambda match: sum(
                binding is None for binding in match.param_bindings or ()
            )
        )
        bound_ops.append(matches[0])

    return bound_ops, warnings


def _snake_to_pascal(s: str) -> str:
    return "".join(p.capitalize() for p in s.split("_"))


_OP_NAMESPACE_PREFIXES = ("special", "linalg", "fft")


def _is_inplace_aten_name(name: str) -> bool:
    """Return whether `name` is an ATen in-place operator name."""

    return name.endswith("_") and not name.endswith("__")


def _public_op_name(aten_name: str) -> str:
    """Map ATen-only spelling to stable InfiniOps public names."""

    public_name = aten_name
    file_prefixes = []

    if public_name.startswith("_"):
        file_prefixes.append("internal")
        public_name = public_name.lstrip("_")

    if _is_inplace_aten_name(public_name):
        public_name = public_name[:-1]

    return "_".join(file_prefixes + [public_name])


def _op_namespace_parts(name: str) -> tuple[str, ...]:
    parts = name.split("_")
    namespaces = []

    if parts and parts[0] == "internal":
        namespaces.append(parts.pop(0))

    if parts and parts[0] in _OP_NAMESPACE_PREFIXES:
        namespaces.append(parts.pop(0))

    return tuple(namespaces)


def _op_class_stem(name: str) -> str:
    parts = name.split("_")

    if parts and parts[0] == "internal":
        parts.pop(0)

    if parts and parts[0] in _OP_NAMESPACE_PREFIXES:
        parts.pop(0)

    return "_".join(parts)


def _op_class_name(name: str) -> str:
    return _snake_to_pascal(_op_class_stem(name))


def _op_cpp_type(name: str) -> str:
    parts = (*_op_namespace_parts(name), _op_class_name(name))

    return "::".join(parts)


def _op_namespace_open(name: str) -> str:
    parts = _op_namespace_parts(name)

    if not parts:
        return "namespace infini::ops {"

    return f"namespace infini::ops::{'::'.join(parts)} {{"


def _op_namespace_close(name: str) -> str:
    parts = _op_namespace_parts(name)

    if not parts:
        return "}  // namespace infini::ops"

    return f"}}  // namespace infini::ops::{'::'.join(parts)}"


def _base_path(op_name: str) -> pathlib.Path:
    return _BASE_DIR / f"{op_name}.h"


def _is_omittable_param(param: Param) -> bool:
    return param.aten_type in _NULLOPT_BY_TYPE or param.default is not None


def _default_call_value(param: Param) -> str:
    if param.aten_type in _NULLOPT_BY_TYPE:
        return _NULLOPT_BY_TYPE[param.aten_type]

    if param.default is not None:
        return _translate_default(param)

    return param.hidden_value()


def _load_aten_entries(version: str) -> list[dict]:
    """Return parsed ATen schemas bundled with installed `torchgen`.

    `WITH_TORCH=ON` already requires a local PyTorch installation. PyTorch
    wheels ship the matching ATen schema under `torchgen/packaged`, including
    vendor forks, so codegen should not depend on fetching PyTorch sources from
    GitHub during CI builds. The official parser expands schemas declared via
    `autogen`, such as `relu.out`, before InfiniOps filters for out variants.
    """

    try:
        packaged, expanded_autogen = _load_packaged_aten_entries()
    except RuntimeError as error:
        raise RuntimeError(f"{error} for PyTorch schema {version!r}") from error

    print(
        "using installed `torchgen` ATen schemas "
        f"for PyTorch schema {version} "
        f"({'with' if expanded_autogen else 'without'} `autogen` expansion).",
        file=sys.stderr,
    )

    return packaged


def _load_packaged_aten_entries() -> tuple[list[dict], bool]:
    """Parse the ATen schemas bundled with installed `torchgen`.

    PyTorch wheels install `torchgen/packaged/ATen/native/native_functions.yaml`;
    using it lets offline CI images generate wrappers against the exact schema
    shipped with their installed torch fork.
    """

    spec = importlib.util.find_spec("torchgen")

    if spec is None:
        raise RuntimeError("could not find the installed `torchgen` package")

    if spec.submodule_search_locations is None:
        raise RuntimeError("installed `torchgen` is not a package")

    for root in spec.submodule_search_locations:
        native_dir = pathlib.Path(root) / "packaged" / "ATen" / "native"
        native_path = native_dir / "native_functions.yaml"
        tags_path = native_dir / "tags.yaml"

        if not native_path.is_file():
            continue

        if not tags_path.is_file():
            print(
                "warning: could not expand ATen `autogen` schemas because "
                f"`{tags_path}` is missing; using explicit schemas only.",
                file=sys.stderr,
            )

            return _load_explicit_aten_entries(native_path), False

        try:
            from torchgen.gen import parse_native_yaml
        except (ImportError, AttributeError) as error:
            print(
                "warning: could not expand ATen `autogen` schemas because "
                f"`torchgen.gen.parse_native_yaml` is unavailable ({error}); "
                "using explicit schemas only.",
                file=sys.stderr,
            )

            return _load_explicit_aten_entries(native_path), False

        try:
            parsed = parse_native_yaml(str(native_path), str(tags_path))
        except Exception as error:
            print(
                "warning: could not expand ATen `autogen` schemas from "
                f"`{native_path}` ({type(error).__name__}: {error}); "
                "using explicit schemas only.",
                file=sys.stderr,
            )

            return _load_explicit_aten_entries(native_path), False

        return (
            [{"func": str(function.func)} for function in parsed.native_functions],
            True,
        )

    raise RuntimeError(
        "could not find installed `torchgen` packaged `native_functions.yaml`"
    )


def _load_explicit_aten_entries(native_path: pathlib.Path) -> list[dict]:
    entries = yaml.safe_load(native_path.read_text())

    if not isinstance(entries, list):
        raise RuntimeError(
            f"expected a list of ATen schemas in `{native_path}`, "
            f"got {type(entries).__name__}"
        )

    return entries


def _find_out_entries(entries: list[dict], op_name: str) -> list[dict]:
    """Return all out-variant entries for `op_name`, with the bare
    `<name>.out(` form first and overload-suffixed variants
    (e.g. `pow.Tensor_Tensor_out(`, `kthvalue.values(`) after.  An
    entry counts as an out-variant when it (a) is named
    `<op_name>.out`, (b) ends in `_out`, or (c) carries a
    `Tensor(<letter>!)` mutability annotation — that last case covers
    multi-output ops named after their output tensors
    (`kthvalue.values`, `mode.values`, …)."""
    bare_prefix = f"{op_name}.out("
    op_overload = re.compile(rf"^{re.escape(op_name)}\.\w+\(")
    mut_tensor = re.compile(r"Tensor\([a-z]!\)")
    bare: list[dict] = []
    others: list[dict] = []

    for entry in entries:
        func = entry.get("func", "")

        if func.startswith(bare_prefix):
            bare.append(entry)
        elif op_overload.match(func) and (
            func.split("(", 1)[0].endswith("_out") or mut_tensor.search(func)
        ):
            others.append(entry)

    return bare + others


def _format_signature(op: Op, *, include_defaults: bool = False) -> str:
    parts = []

    for param in op.visible_params:
        prefix = "" if param.is_mutable_tensor else "const "
        text = f"{prefix}{param.cpp_type} {param.api_name}"

        if include_defaults and param.default is not None:
            text += f" = {_translate_default(param)}"

        parts.append(text)

    return ", ".join(parts)


def _visible_signature_key(op: Op) -> tuple[str, ...]:
    """C++ overload identity for the user-facing API.

    Parameter names and top-level `const` do not distinguish C++ overloads, so
    only the exposed C++ type sequence participates in duplicate detection.
    """

    return tuple(param.cpp_type for param in op.visible_params)


def _canonical_overload_score(index: int, op: Op) -> tuple[bool, int, int, str, int]:
    """Sort key for duplicate visible signatures.

    Prefer the canonical unsuffixed InfiniOps name, then the schema that hides
    fewer ATen-only defaults, then the shorter deterministic name.
    """

    return (
        op.infini_name != op.aten_name,
        sum(param.is_hidden for param in op.params),
        len(op.infini_name),
        op.infini_name,
        index,
    )


def _dedupe_visible_overloads(ops: list[Op]) -> tuple[list[Op], list[tuple[Op, Op]]]:
    """Drop overloads that collapse to the same visible C++ signature.

    Returns the selected overloads in the original schema order plus a list of
    `(skipped, kept)` duplicate pairs for diagnostics.
    """
    winners: dict[tuple[str, ...], tuple[int, Op]] = {}
    duplicates: list[tuple[Op, tuple[str, ...]]] = []

    for index, op in enumerate(ops):
        key = _visible_signature_key(op)
        current = winners.get(key)

        if current is None:
            winners[key] = (index, op)
            continue

        current_index, current_op = current

        if _canonical_overload_score(index, op) < _canonical_overload_score(
            current_index, current_op
        ):
            duplicates.append((current_op, key))
            winners[key] = (index, op)
        else:
            duplicates.append((op, key))

    selected_indices = {index for index, _ in winners.values()}
    selected = [op for index, op in enumerate(ops) if index in selected_indices]
    duplicate_pairs = [
        (skipped, winners[key][1])
        for skipped, key in duplicates
        if winners[key][1] is not skipped
    ]

    return selected, duplicate_pairs


def _translate_default(param: Param) -> str:
    """Translate a YAML default literal to a C++ literal."""
    raw = param.default

    if raw == "True":
        return "true"

    if raw == "False":
        return "false"

    if raw == "None":
        return "{}"

    return raw  # numeric literals (`0`, `1`, `1.0`) pass through


def _generate_base_includes(ops: list[Op]) -> str:
    cpp_types = [param.cpp_type for op in ops for param in op.visible_params]
    includes = []

    if any("std::optional" in cpp_type for cpp_type in cpp_types):
        includes.append("#include <optional>")

    if any("std::string" in cpp_type for cpp_type in cpp_types):
        includes.append("#include <string>")

    if any("std::vector" in cpp_type for cpp_type in cpp_types):
        includes.append("#include <vector>")

    includes.append('#include "operator.h"')

    return "\n".join(includes)


def _generate_base_header(name: str, ops: list[Op]) -> str:
    pascal = _op_class_name(name)

    member_decls = []
    tensor_member_order = []
    seen_tensor_members = set()
    optional_tensor_member_order = []
    seen_optional_tensor_members = set()
    scalar_member_order = []
    scalar_member_types = {}

    for op in ops:
        for param in op.tensor_params:
            api_name = param.api_name

            if api_name in seen_tensor_members:
                continue

            seen_tensor_members.add(api_name)
            tensor_member_order.append(api_name)
            member_decls.append(f"  Tensor::Shape {api_name}_shape_;")
            member_decls.append(f"  Tensor::Strides {api_name}_strides_;")
            member_decls.append(f"  DataType {api_name}_type_;")

        for param in op.visible_params:
            api_name = param.api_name

            if not param.is_optional_tensor or api_name in seen_optional_tensor_members:
                continue

            seen_optional_tensor_members.add(api_name)
            optional_tensor_member_order.append(api_name)
            member_decls.append(f"  bool has_{api_name}_{{false}};")
            member_decls.append(f"  Tensor::Shape {api_name}_shape_;")
            member_decls.append(f"  Tensor::Strides {api_name}_strides_;")
            member_decls.append(f"  DataType {api_name}_type_{{DataType::kFloat32}};")

        # Visible non-tensor params (scalars, strings, vectors) are also
        # stored on the base so backends can dispatch on them later — not
        # only at the moment `operator()` is invoked. Reviewers flagged
        # this on multiple PRs (e.g. `n` on
        # `special_chebyshev_polynomial_v_n_scalar`). Same-named params
        # across overloads must share a type; if they conflict, the second
        # overload's member is dropped (later constructors leave it
        # default-initialised).
        for param in op.visible_params:
            api_name = param.api_name

            if (
                param.is_tensor
                or param.is_optional_tensor
                or api_name in scalar_member_types
            ):
                continue

            scalar_member_order.append(api_name)
            scalar_member_types[api_name] = param.cpp_type
            member_decls.append(f"  {param.cpp_type} {api_name}_{{}};")

    member_decls.append("  int device_index_{0};")

    constructors = []
    calls = []

    for op in ops:
        init_pieces = []
        tensor_params = {param.api_name: param for param in op.tensor_params}
        optional_tensor_params = {
            param.api_name: param
            for param in op.visible_params
            if param.is_optional_tensor
        }
        scalar_params = {
            param.api_name: param
            for param in op.visible_params
            if not param.is_tensor
            and not param.is_optional_tensor
            and scalar_member_types.get(param.api_name) == param.cpp_type
        }

        for param_name in tensor_member_order:
            param = tensor_params.get(param_name)

            if param is None:
                continue

            api_name = param.api_name
            init_pieces.append(f"        {api_name}_shape_{{{api_name}.shape()}}")
            init_pieces.append(f"        {api_name}_strides_{{{api_name}.strides()}}")
            init_pieces.append(f"        {api_name}_type_{{{api_name}.dtype()}}")

        for param_name in optional_tensor_member_order:
            param = optional_tensor_params.get(param_name)

            if param is None:
                continue

            api_name = param.api_name
            init_pieces.append(f"        has_{api_name}_{{{api_name}.has_value()}}")
            init_pieces.append(
                f"        {api_name}_shape_{{{api_name} ? "
                f"Tensor::Shape{{{api_name}->shape()}} : Tensor::Shape{{}}}}"
            )
            init_pieces.append(
                f"        {api_name}_strides_{{{api_name} ? "
                f"Tensor::Strides{{{api_name}->strides()}} : Tensor::Strides{{}}}}"
            )
            init_pieces.append(
                f"        {api_name}_type_{{{api_name} ? "
                f"{api_name}->dtype() : DataType::kFloat32}}"
            )

        for param_name in scalar_member_order:
            param = scalar_params.get(param_name)

            if param is None:
                continue

            api_name = param.api_name
            init_pieces.append(f"        {api_name}_{{{api_name}}}")

        # All out tensors share a device; use the first one. Keep this last
        # so initializer order follows the member declaration order.
        init_pieces.append(
            f"        device_index_{{{op.out_params[0].api_name}.device().index()}}"
        )

        init_list = ",\n".join(init_pieces).lstrip()
        constructors.append(
            f"  {pascal}({_format_signature(op)})\n      : {init_list} {{}}"
        )
        calls.append(f"  virtual void operator()({_format_signature(op)}) const = 0;")

    return _BASE_TEMPLATE.format(
        name_uc=name.upper(),
        pascal=pascal,
        namespace_open=_op_namespace_open(name),
        namespace_close=_op_namespace_close(name),
        includes=_generate_base_includes(ops),
        constructors="\n\n".join(constructors),
        op_calls="\n\n".join(calls),
        member_decls="\n\n".join(member_decls),
    )


def _generate_torch_header(name: str, ops: list[Op]) -> str:
    pascal = _op_class_name(name)
    op_type = _op_cpp_type(name)
    op_calls = "\n\n".join(
        f"  void operator()({_format_signature(op)}) const override;" for op in ops
    )

    return _TORCH_HEADER_TEMPLATE.format(
        name_uc=name.upper(),
        name=name,
        pascal=pascal,
        op_type=op_type,
        op_calls=op_calls,
        slot=_PYTORCH_SLOT,
    )


def _generate_torch_method_source(name: str, op: Op) -> str:
    op_type = _op_cpp_type(name)
    conversion_lines = []
    out_device_index = f"{op.out_params[0].api_name}.device().index()"

    def _optional_aten_type(param: Param) -> str:
        return _NULLOPT_BY_TYPE[param.aten_type].removesuffix("{}")

    def _optional_aten_value(schema_param: Param, api_param: Param) -> str:
        api_name = api_param.api_name

        if schema_param.aten_type == "Tensor?":
            data_expr = f"const_cast<void*>({api_name}->data())"

            return (
                f"ToAtenTensor<kDev>({data_expr}, {api_name}->shape(), "
                f"{api_name}->strides(), {api_name}->dtype(), "
                f"device_index)"
            )

        if schema_param.aten_type == "Scalar?":
            return f"at::Scalar(*{api_name})"

        if schema_param.aten_type == "ScalarType?":
            return f"ToAtenDataType(*{api_name})"

        if schema_param.aten_type == "str?":
            return f"c10::string_view(*{api_name})"

        if schema_param.aten_type.startswith(
            ("int[", "SymInt[")
        ) or schema_param.aten_type in {
            "int[]?",
            "SymInt[]?",
        }:
            return f"at::IntArrayRef(*{api_name})"

        if schema_param.aten_type == "float[]?":
            return f"at::ArrayRef<double>(*{api_name})"

        return f"*{api_name}"

    def _append_optional_conversion(schema_param: Param, api_param: Param) -> None:
        api_name = api_param.api_name
        optional_type = _optional_aten_type(schema_param)
        conversion_lines.append(f"  {optional_type} at_{schema_param.name};")
        conversion_lines.append(f"  if ({api_name}.has_value()) {{")
        conversion_lines.append(
            f"    at_{schema_param.name} = "
            f"{optional_type}{{{_optional_aten_value(schema_param, api_param)}}};"
        )
        conversion_lines.append("  }")

    for schema_index, param in enumerate(op.params):
        if (
            not param.is_tensor
            and not param.is_tensor_list
            and not param.is_optional_tensor_list
        ):
            continue

        api_param = op.api_param_for(schema_index)

        if api_param is None:
            continue

        api_name = api_param.api_name

        if param.is_optional_tensor_list:
            conversion_lines.append(
                f"  c10::List<c10::optional<at::Tensor>> at_{param.name};"
            )
            conversion_lines.append(f"  at_{param.name}.reserve({api_name}.size());")
            conversion_lines.append(f"  for (const auto& tensor : {api_name}) {{")
            conversion_lines.append("    if (tensor.has_value()) {")
            conversion_lines.append(
                "      at_"
                f"{param.name}.push_back(ToAtenTensor<kDev>("
                "const_cast<void*>(tensor->data()), tensor->shape(), "
                "tensor->strides(), tensor->dtype(), device_index));"
            )
            conversion_lines.append("    } else {")
            conversion_lines.append(f"      at_{param.name}.push_back(c10::nullopt);")
            conversion_lines.append("    }")
            conversion_lines.append("  }")
            continue

        if param.is_tensor_list:
            conversion_lines.append(f"  std::vector<at::Tensor> at_{param.name};")
            conversion_lines.append(f"  at_{param.name}.reserve({api_name}.size());")
            conversion_lines.append(f"  for (const auto& tensor : {api_name}) {{")
            conversion_lines.append(
                "    at_"
                f"{param.name}.push_back(ToAtenTensor<kDev>("
                "const_cast<void*>(tensor.data()), tensor.shape(), tensor.strides(), "
                "tensor.dtype(), device_index));"
            )
            conversion_lines.append("  }")
            continue

        use_mutable_data = param.is_mutable_tensor and not (
            op.explicit_inplace_out is not None and schema_index == 0
        )
        data_expr = (
            f"{api_name}.data()"
            if use_mutable_data
            else f"const_cast<void*>({api_name}.data())"
        )
        conversion_lines.append(
            f"  auto at_{param.name} = ToAtenTensor<kDev>(\n"
            f"      {data_expr}, {api_name}_shape_, {api_name}_strides_,\n"
            f"      {api_name}_type_, device_index);"
        )

    if op.explicit_inplace_out is not None:
        out_name = op.explicit_inplace_out.api_name
        conversion_lines.append(
            f"  auto at_{out_name} = ToAtenTensor<kDev>(\n"
            f"      {out_name}.data(), {out_name}_shape_, {out_name}_strides_,\n"
            f"      {out_name}_type_, device_index);"
        )

    for schema_index, param in enumerate(op.params):
        api_param = op.api_param_for(schema_index)

        if (
            api_param is not None
            and param.aten_type in _EXPOSED_OPTIONAL_CPP_TYPES
            and not param.is_optional_tensor_list
        ):
            _append_optional_conversion(param, api_param)

    def _render_arg(schema_index, p):
        api_param = op.api_param_for(schema_index)

        if api_param is None:
            return _default_call_value(p)

        if p.is_hidden:
            return _default_call_value(p)

        if p.aten_type in _EXPOSED_OPTIONAL_CPP_TYPES:
            return f"at_{p.name}"

        if p.is_tensor or p.is_tensor_list:
            return f"at_{p.name}"

        if p.aten_type == "ScalarType":
            return f"ToAtenDataType({api_param.api_name})"

        return api_param.api_name

    if op.is_inplace:
        # In-place ATen calls keep the mutable input in positional order,
        # unlike `_out` calls which place output tensors first.
        input_param = op.params[0]
        arg_order = op.params[1:]
        rendered_args = ", ".join(
            _render_arg(index + 1, p) for index, p in enumerate(arg_order)
        )
        if op.explicit_inplace_out is not None:
            out_name = op.explicit_inplace_out.api_name
            aten_call = f"at_{out_name}.copy_(at_{input_param.name});\n"
            aten_call += f"  at_{out_name}.{op.aten_name}({rendered_args})"
        else:
            aten_call = f"at_{input_param.name}.{op.aten_name}({rendered_args})"
    else:
        # ATen `_out` form puts all out tensors first, then non-out params
        # in YAML order. Hardcoded-nullopt params become `at::nullopt`.
        arg_order = [(index, p) for index, p in enumerate(op.params) if p.is_out] + [
            (index, p) for index, p in enumerate(op.params) if not p.is_out
        ]
        rendered_args = ", ".join(_render_arg(index, p) for index, p in arg_order)
        aten_call = f"at::{op.aten_name}_out({rendered_args})"

    return _TORCH_METHOD_TEMPLATE.format(
        op_type=op_type,
        op_call_signature=_format_signature(op),
        tensor_conversions="\n".join(conversion_lines),
        out_device_index=out_device_index,
        c10_condition=" ||\n      ".join(
            f"kDev == Device::Type::{dev}" for dev in _C10_DEVICE_TYPES
        ),
        # The generated call expression resolves the right kernel via C++
        # overload resolution from the argument types we pass.
        aten_call=aten_call,
        slot=_PYTORCH_SLOT,
    )


def _generate_torch_source(name: str, ops: list[Op]) -> str:
    op_type = _op_cpp_type(name)
    methods = "\n\n".join(_generate_torch_method_source(name, op) for op in ops)
    c10_includes = "\n".join(
        f"#ifdef WITH_{dev.removeprefix('k').upper()}\n"
        f'#include "torch/{dev.removeprefix("k").lower()}/c10.h"\n'
        f"#endif"
        for dev in _C10_DEVICE_TYPES
    )
    # Guard each explicit instantiation by the matching `WITH_<DEV>` macro
    # so a build that only enables a subset of devices does not pay the
    # ATen template-instantiation cost (and memory pressure) for the
    # devices it does not link against. Each macro is set by
    # `target_compile_definitions` in `src/CMakeLists.txt`.
    instantiations = "\n".join(
        f"#ifdef WITH_{dev.removeprefix('k').upper()}\n"
        f"template class Operator<{op_type}, Device::Type::{dev}, {_PYTORCH_SLOT}>;\n"
        f"#endif"
        for dev in _DEVICE_TYPES
    )

    return _TORCH_SOURCE_TEMPLATE.format(
        name=name,
        methods=methods,
        c10_includes=c10_includes,
        instantiations=instantiations,
    )


_BASE_TEMPLATE = """\
#ifndef INFINI_OPS_BASE_{name_uc}_H_
#define INFINI_OPS_BASE_{name_uc}_H_

{includes}

{namespace_open}

class {pascal} : public Operator<{pascal}> {{
 public:
{constructors}

{op_calls}

 protected:
{member_decls}
}};

{namespace_close}

#endif
"""


_TORCH_HEADER_TEMPLATE = """\
#ifndef INFINI_OPS_TORCH_{name_uc}_H_
#define INFINI_OPS_TORCH_{name_uc}_H_

#include "base/{name}.h"

namespace infini::ops {{

template <Device::Type kDev>
class Operator<{op_type}, kDev, {slot}> : public {op_type} {{
 public:
  using {op_type}::{pascal};

{op_calls}
}};

}}  // namespace infini::ops

#endif
"""


_TORCH_METHOD_TEMPLATE = """\
template <Device::Type kDev>
void Operator<{op_type}, kDev, {slot}>::operator()({op_call_signature}) const {{
  const auto device_index = {out_device_index};
  const auto run = [&] {{
{tensor_conversions}

    {aten_call};
  }};

  if constexpr ({c10_condition}) {{
    const typename C10<kDev>::StreamGuard stream_guard{{
        C10<kDev>::GetStreamFromExternal(stream_, device_index)}};
    run();
    return;
  }}

  run();
}}
"""


_TORCH_SOURCE_TEMPLATE = """\
#include "torch/{name}/{name}.h"

#include "torch/c10.h"
{c10_includes}
#include "torch/tensor_.h"

namespace infini::ops {{

{methods}

{instantiations}

}}  // namespace infini::ops
"""


def _find_clang_format() -> str | None:
    """Return the path to a `clang-format` binary, or `None` if unavailable.

    Generated files live under `generated/`, which is gitignored, so offline CI
    containers should not block or try network access just to format them.
    """

    found = shutil.which("clang-format")

    if found:
        return found

    print(
        "`clang-format` not found on PATH; generated files will be emitted "
        "without formatting.",
        file=sys.stderr,
    )

    return None


def _clang_format(text: str, path: pathlib.Path) -> str:
    """Pipe `text` through `clang-format` so generated headers / sources
    satisfy the same style check (`clang-format` v21) that CI runs.
    `path` informs include sorting (the file's own header should come
    first in a `.cc`).  If no `clang-format` binary is available, return
    the input unchanged."""

    if _CLANG_FORMAT is None:
        return text

    return subprocess.run(
        [_CLANG_FORMAT, f"--assume-filename={path}"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def _emit(name: str, ops: list[Op], *, emit_base: bool) -> set[pathlib.Path]:
    base_path = _GENERATED_BASE_DIR / f"{name}.h"
    torch_dir = _GENERATED_TORCH_DIR / name
    torch_header_path = torch_dir / f"{name}.h"
    torch_source_path = torch_dir / f"{name}.cc"
    emitted_paths = set()

    if emit_base:
        _write_text_if_changed(
            base_path, _clang_format(_generate_base_header(name, ops), base_path)
        )
        emitted_paths.add(base_path)

    _write_text_if_changed(
        torch_header_path,
        _clang_format(_generate_torch_header(name, ops), torch_header_path),
    )
    _write_text_if_changed(
        torch_source_path,
        _clang_format(_generate_torch_source(name, ops), torch_source_path),
    )
    emitted_paths.add(torch_header_path)
    emitted_paths.add(torch_source_path)

    return emitted_paths


def _select_op_names(cli_ops, default_ops, config):
    if config is None:
        return cli_ops or default_ops

    aten_names_by_public_name = collections.defaultdict(list)
    for op_name in default_ops:
        aten_names_by_public_name[_public_op_name(op_name)].append(op_name)

    selected_public_names = ops_config.torch_op_names(
        config, aten_names_by_public_name, _PYTORCH_SLOT
    )
    selected = [
        aten_name
        for public_name in selected_public_names
        for aten_name in aten_names_by_public_name.get(public_name, (public_name,))
    ]

    if cli_ops:
        allowed = set(cli_ops)
        selected = [op_name for op_name in selected if op_name in allowed]

    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ops",
        nargs="*",
        help="Override the op allowlist. If omitted, reads `scripts/torch_ops.yaml`.",
    )
    parser.add_argument(
        "--ops-config",
        type=pathlib.Path,
        help="Path to an `ops.json` operator and implementation selection.",
    )
    parser.add_argument(
        "--pytorch-version",
        default=os.environ.get("INFINI_OPS_PYTORCH_VERSION", _DEFAULT_PYTORCH_VERSION),
        help=(
            "PyTorch version label used in diagnostics. Schemas are read from "
            "the locally installed `torchgen` package. Default: `%(default)s`. "
            "Can also be set via the `INFINI_OPS_PYTORCH_VERSION` environment "
            "variable."
        ),
    )
    args = parser.parse_args()

    global _CLANG_FORMAT
    _CLANG_FORMAT = _find_clang_format()

    default_ops = yaml.safe_load(_OPS_YAML_PATH.read_text())
    config = ops_config.load_ops_config(args.ops_config) if args.ops_config else None
    op_names = _select_op_names(args.ops, default_ops, config)
    aten_entries = _load_aten_entries(args.pytorch_version)

    skipped: list[tuple[str, str]] = []
    metadata: list[dict] = []
    expected_base_files: set[pathlib.Path] = set()
    expected_torch_files: set[pathlib.Path] = set()
    ops_by_public_name: dict[str, list[Op]] = collections.defaultdict(list)

    for name in op_names:
        candidates = _find_out_entries(aten_entries, name)

        if not candidates:
            skipped.append((name, f"no `.out` variant for `{name}` in YAML"))
            continue

        usable: list[Op] = []
        last_reason = ""

        for entry in candidates:
            try:
                op = _parse_func(entry["func"])

                for param in op.params:
                    _ = param.cpp_type  # eagerly raise on unsupported types
            except (NotImplementedError, ValueError) as exc:
                last_reason = str(exc)
                continue

            if not op.is_testable:
                last_reason = "no testable tensor input/output pair"
                continue

            usable.append(op)

        if not usable:
            skipped.append((name, last_reason or "no usable overload"))
            continue

        usable, duplicate_overloads = _dedupe_visible_overloads(usable)

        for skipped_op, kept_op in duplicate_overloads:
            skipped.append(
                (
                    skipped_op.infini_name,
                    "duplicate visible C++ signature for "
                    f"`{name}`; using `{kept_op.infini_name}`",
                )
            )

        ops_by_public_name[usable[0].infini_name].extend(usable)

    for public_name, usable in ops_by_public_name.items():
        usable, duplicate_overloads = _dedupe_visible_overloads(usable)

        for skipped_op, kept_op in duplicate_overloads:
            skipped.append(
                (
                    skipped_op.infini_name,
                    "duplicate visible C++ signature for "
                    f"`{public_name}`; using `{kept_op.infini_name}`",
                )
            )

        base_exists = _base_path(public_name).exists()

        if base_exists:
            usable, base_warnings = _bind_existing_base_overloads(public_name, usable)

            for warning in base_warnings:
                skipped.append((public_name, warning))

            if not usable:
                skipped.append(
                    (
                        public_name,
                        "existing base has no overload compatible with ATen schema",
                    )
                )
                continue

        # Emit one InfiniOps wrapper per ATen op. Distinct visible overloads
        # become overloaded constructors / `operator()` methods on the same
        # class (`Pow` exposes both tensor and scalar exponents). Overloads
        # that collapse to the same C++ signature after hidden defaults are
        # skipped above. When a hand-written `src/base/<name>.h` exists,
        # skip emitting `generated/base/<name>.h` so the hand-written one
        # wins (the generated torch source's `#include "base/<name>.h"`
        # resolves through `src/` first). Signature mismatches surface as
        # compile errors with a clear message — drop the op from the YAML
        # to suppress.
        emitted_paths = _emit(public_name, usable, emit_base=not base_exists)
        expected_base_files.update(
            path for path in emitted_paths if path.is_relative_to(_GENERATED_BASE_DIR)
        )
        expected_torch_files.update(
            path for path in emitted_paths if path.is_relative_to(_GENERATED_TORCH_DIR)
        )

        for op in usable:
            metadata.append(
                {
                    "name": public_name,
                    "aten_name": op.aten_name,
                    "overload_name": op.infini_name,
                    "params": [
                        {
                            "name": p.api_name,
                            "type": p.aten_type,
                            "is_tensor": p.is_tensor,
                            "is_out": p.is_out,
                            "schema_index": next(
                                (
                                    schema_index
                                    for schema_index in range(len(op.params))
                                    if op.api_param_for(schema_index) is p
                                ),
                                None,
                            ),
                        }
                        for p in op.visible_params
                    ],
                }
            )

    _GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    _write_text_if_changed(
        _METADATA_PATH, json.dumps({"ops": metadata}, indent=2) + "\n"
    )
    _remove_stale_files(_GENERATED_BASE_DIR, expected_base_files)
    _remove_stale_files(_GENERATED_TORCH_DIR, expected_torch_files)

    generated_names = sorted({m["name"] for m in metadata})
    print(
        f"generated {len(metadata)} overloads across {len(generated_names)} ops: "
        f"{generated_names}"
    )

    for name, reason in skipped:
        print(f"  skipped {name!r}: {reason}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
