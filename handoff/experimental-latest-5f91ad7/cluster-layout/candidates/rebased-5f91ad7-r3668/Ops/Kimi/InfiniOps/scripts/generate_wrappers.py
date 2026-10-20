import argparse
import concurrent.futures
import dataclasses
import functools
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ops_config  # noqa: E402

try:
    import clang.cindex
    from clang.cindex import CursorKind
except ImportError:
    clang = None
    CursorKind = None

_SRC_DIR = pathlib.Path("src")

_BASE_DIR = _SRC_DIR / "base"

_GENERATION_DIR = pathlib.Path("generated")

# Base headers emitted by `generate_torch_ops.py` live alongside the
# hand-written ones in `src/base/`, but in a parallel tree under
# `generated/base/` so they are not committed.
_GENERATED_BASE_DIR = _GENERATION_DIR / "base"

_BINDINGS_DIR = _GENERATION_DIR / "bindings"

_GENERATED_SRC_DIR = _GENERATION_DIR / "src"

_INCLUDE_DIR = _GENERATION_DIR / "include"

_INDENTATION = "  "

_OP_NAMESPACE_PREFIXES = ("special", "linalg", "fft")


def _op_namespace_parts(op_name):
    parts = op_name.split("_")
    namespaces = []

    if parts and parts[0] == "internal":
        namespaces.append(parts.pop(0))

    if parts and parts[0] in _OP_NAMESPACE_PREFIXES:
        namespaces.append(parts.pop(0))

    return tuple(namespaces)


def _op_class_stem(op_name):
    parts = op_name.split("_")

    if parts and parts[0] == "internal":
        parts.pop(0)

    if parts and parts[0] in _OP_NAMESPACE_PREFIXES:
        parts.pop(0)

    return "_".join(parts)


def _op_class_name(op_name):
    return _snake_to_pascal(_op_class_stem(op_name))


def _op_symbol_name(op_name):
    return _snake_to_pascal(op_name)


def _op_cpp_type(op_name):
    parts = (*_op_namespace_parts(op_name), _op_class_name(op_name))

    return "::infini::ops::" + "::".join(parts)


def _op_relative_type(op_name):
    parts = (*_op_namespace_parts(op_name), _op_class_name(op_name))

    return "::".join(parts)


def _get_infini_rt_include_flags():
    include_dirs = []

    for include_dir in os.environ.get("INFINI_RT_INCLUDE_DIRS", "").split(os.pathsep):
        if include_dir:
            include_dirs.append(pathlib.Path(include_dir))

    infini_rt_root = os.environ.get("INFINI_RT_ROOT")
    if infini_rt_root:
        include_dirs.append(pathlib.Path(infini_rt_root) / "include")

    infini_rt_source_dir = os.environ.get("INFINI_RT_SOURCE_DIR")
    if infini_rt_source_dir:
        infini_rt_source_path = pathlib.Path(infini_rt_source_dir)
        include_dirs.extend(
            (
                infini_rt_source_path / "include",
                infini_rt_source_path / "generated" / "include",
            )
        )

    flags = []
    seen = set()
    for include_dir in include_dirs:
        include_dir = include_dir.resolve()
        if include_dir.exists() and include_dir not in seen:
            flags.extend(("-I", str(include_dir)))
            seen.add(include_dir)

    return tuple(flags)


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


@functools.lru_cache(maxsize=1)
def _get_system_include_flags():
    """Probe the system C++ compiler for default include paths so libclang
    can resolve standard headers when parsing an op's base header."""
    compilers = []

    for compiler in ("clang++", "g++"):
        if shutil.which(compiler) is not None:
            compilers.append(compiler)

    system_include_flags = []

    for compiler in compilers:
        for line in subprocess.getoutput(
            f"{compiler} -E -x c++ -v /dev/null"
        ).splitlines():
            if not line.startswith(" "):
                continue

            system_include_flags.append("-isystem")
            system_include_flags.append(line.strip())

    return tuple(system_include_flags)


def _find_base_header(op_name):
    """Resolve the base header for `op_name`, preferring the hand-written
    `src/base/<op>.h` over the auto-generated `generated/base/<op>.h`.
    Mirrors the include-path resolution order used at compile time."""
    src_path = _BASE_DIR / f"{op_name}.h"

    if src_path.exists():
        return src_path

    generated_path = _GENERATED_BASE_DIR / f"{op_name}.h"

    if generated_path.exists():
        return generated_path

    raise FileNotFoundError(f"no base header for op {op_name!r}")


class _ParsedType:
    def __init__(self, spelling):
        self.spelling = spelling


class _ParsedArgument:
    def __init__(self, type_spelling, spelling):
        self.type = _ParsedType(type_spelling)
        self.spelling = spelling


class _ParsedFunction:
    def __init__(self, arguments):
        self._arguments = arguments

    def get_arguments(self):
        return self._arguments


class _OperatorExtractor:
    def __call__(self, op_name):
        if clang is None:
            return _parse_operator_header(op_name)

        index = clang.cindex.Index.create()
        args = (
            (
                "-std=c++17",
                "-x",
                "c++",
                "-I",
                "src",
                "-I",
                str(_GENERATION_DIR),
            )
            + _get_infini_rt_include_flags()
            + _get_system_include_flags()
        )
        translation_unit = index.parse(str(_find_base_header(op_name)), args=args)

        nodes = tuple(type(self)._find(translation_unit.cursor, op_name))

        constructors = []
        calls = []

        for node in nodes:
            if node.kind == CursorKind.CONSTRUCTOR:
                constructors.append(node)
            elif node.kind == CursorKind.CXX_METHOD and node.spelling == "operator()":
                calls.append(node)

        parsed_operator = _parse_operator_header(op_name)
        constructors = _prefer_header_type_spellings(
            constructors, parsed_operator.constructors
        )
        calls = _prefer_header_type_spellings(calls, parsed_operator.calls)

        return _Operator(op_name, constructors, calls)

    @staticmethod
    def _find(node, op_name):
        class_name = _op_class_name(op_name)

        if node.semantic_parent and node.semantic_parent.spelling == class_name:
            yield node

        for child in node.get_children():
            yield from _OperatorExtractor._find(child, op_name)


def _parse_operator_header(op_name):
    class_name = _op_class_name(op_name)
    source = _strip_cpp_comments(_find_base_header(op_name).read_text())
    class_body = _extract_class_body(source, class_name)
    constructors = [
        _ParsedFunction(_parse_parameter_list(params))
        for params in _find_signature_parameters(
            class_body, rf"(?:explicit\s+)?{class_name}\s*\("
        )
    ]
    calls = [
        _ParsedFunction(_parse_parameter_list(params))
        for params in _find_signature_parameters(
            class_body, r"(?:virtual\s+)?void\s+operator\s*\(\s*\)\s*\("
        )
    ]

    return _Operator(op_name, constructors, calls)


def _prefer_header_type_spellings(functions, header_functions):
    header_by_names = {}

    for function in header_functions:
        names = tuple(arg.spelling for arg in function.get_arguments())
        header_by_names.setdefault(names, []).append(function)

    aligned = []

    for function in functions:
        names = tuple(arg.spelling for arg in function.get_arguments())
        matches = header_by_names.get(names, ())

        if len(matches) == 1:
            aligned.append(matches[0])
        else:
            aligned.append(function)

    return aligned


def _strip_cpp_comments(source):
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return re.sub(r"//.*", "", source)


def _extract_class_body(source, class_name):
    match = re.search(
        rf"\bclass(?:\s+\[\[.*?\]\])?\s+{class_name}\b[^{{]*{{",
        source,
        flags=re.DOTALL,
    )

    if match is None:
        raise ValueError(f"no class definition for {class_name!r}")

    start = match.end()
    depth = 1
    index = start

    while index < len(source):
        char = source[index]

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]

        index += 1

    raise ValueError(f"unterminated class definition for {class_name!r}")


def _find_signature_parameters(source, pattern):
    params = []

    for match in re.finditer(pattern, source):
        opening_paren = match.end() - 1

        if opening_paren < 0 or source[opening_paren] != "(":
            continue

        closing_paren = _find_matching_delimiter(source, opening_paren, "(", ")")
        params.append(source[opening_paren + 1 : closing_paren])

    return params


def _find_matching_delimiter(source, start, opening, closing):
    depth = 0

    for index in range(start, len(source)):
        char = source[index]

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index

    raise ValueError(f"unmatched delimiter {opening!r}")


def _parse_parameter_list(params):
    arguments = []

    for param in _split_top_level(params, ","):
        param = _strip_default_argument(param.strip())

        if not param or param == "void":
            continue

        match = re.match(r"(.+?[\s*&]+)([A-Za-z_][A-Za-z0-9_]*)$", param)

        if match is None:
            raise ValueError(f"could not parse parameter {param!r}")

        arguments.append(_ParsedArgument(match.group(1).strip(), match.group(2)))

    return arguments


def _split_top_level(text, delimiter):
    parts = []
    start = 0
    depth = 0
    pairs = {"<": ">", "(": ")", "[": "]", "{": "}"}
    closing = {value: key for key, value in pairs.items()}

    for index, char in enumerate(text):
        if char in pairs:
            depth += 1
        elif char in closing:
            depth -= 1
        elif char == delimiter and depth == 0:
            parts.append(text[start:index])
            start = index + 1

    parts.append(text[start:])
    return parts


def _strip_default_argument(param):
    parts = _split_top_level(param, "=")
    return parts[0].strip()


@dataclasses.dataclass(frozen=True)
class _Implementation:
    path: pathlib.Path
    backend: str


class _Operator:
    def __init__(self, name, constructors, calls, implementations=()):
        self.name = name

        self.constructors = constructors

        self.calls = calls

        self.implementations = tuple(implementations)

    def has_backend(self, backend):
        return any(
            implementation.backend == backend for implementation in self.implementations
        )


def _parameter_type_signature(node):
    return tuple(
        " ".join(arg.type.spelling.split())
        for arg in node.get_arguments()
        if arg.spelling != "stream"
    )


def _find_optional_tensor_params(op_name):
    """Return a set of parameter names declared as `std::optional<Tensor>` in
    the base header. `libclang` resolves the type to `int` when the STL
    headers are not fully available, so we fall back to a regex scan of the
    source text.
    """
    source = _find_base_header(op_name).read_text()

    return set(re.findall(r"std::optional<Tensor>\s+(\w+)", source))


def _find_optional_non_tensor_params(op_name):
    """Return parameter names declared as non-Tensor `std::optional`.

    Some generated ATen bases have overloads that reuse a parameter name across
    different optional kinds, e.g. `clamp(..., std::optional<double> min, ...)`
    and `clamp(..., std::optional<Tensor> min, ...)`. The optional-Tensor
    regex fallback is name-based, so record non-Tensor optionals too to avoid
    treating the scalar overload as a Tensor overload.
    """
    source = _find_base_header(op_name).read_text()

    return {
        name
        for cpp_type, name in re.findall(r"std::optional<([^>]+)>\s+(\w+)", source)
        if "Tensor" not in cpp_type
    }


def _find_vector_tensor_params(op_name):
    """Return a set of parameter names declared as `std::vector<Tensor>` in
    the base header.
    """
    source = _find_base_header(op_name).read_text()

    return set(re.findall(r"std::vector\s*<\s*Tensor\s*>\s+(\w+)", source))


def _find_optional_vector_tensor_params(op_name):
    """Return params declared as `std::vector<std::optional<Tensor>>`."""
    source = _find_base_header(op_name).read_text()

    return set(
        re.findall(
            r"std::vector\s*<\s*std::optional\s*<\s*Tensor\s*>\s*>\s+(\w+)",
            source,
        )
    )


def _vector_tensor_kind(arg, vector_params, optional_vector_params):
    """Classify a Tensor vector while preserving optional element types."""
    spelling = "".join(arg.type.spelling.split())
    spelling = spelling.removeprefix("const").removesuffix("&")

    if (
        spelling.startswith("std::vector<")
        and "std::optional<" in spelling
        and ("Tensor" in spelling or "TensorView" in spelling)
    ):
        return "optional"

    if spelling.startswith("std::vector<") and (
        "Tensor" in spelling or "TensorView" in spelling
    ):
        return "vector"

    if arg.spelling in optional_vector_params:
        return "optional"

    if arg.spelling in vector_params:
        return "vector"

    return None


def _find_vector_int64_params(op_name):
    """Return a set of parameter names declared as `std::vector<int64_t>` in
    the base header.

    libclang on systems where the STL headers are not fully indexable
    silently falls back to reporting the type as `int` for these params,
    which then leaks into the generated bindings as `const int padding`
    instead of `const std::vector<int64_t> padding` and breaks the call
    to the base operator.  Regex-scan the source so the binding's
    parameter type comes from the actual declaration.
    """
    source = _find_base_header(op_name).read_text()

    return set(re.findall(r"std::vector<int64_t>\s+(\w+)", source))


def _find_optional_vector_int64_params(op_name):
    """Return params declared as `std::optional<std::vector<int64_t>>`.

    libclang can report these as plain `int`, just like non-optional vector
    params, so scan the base header to preserve the generated API signature.
    """
    source = _find_base_header(op_name).read_text()

    return set(
        re.findall(
            r"std::optional\s*<\s*std::vector\s*<\s*int64_t\s*>\s*>\s+(\w+)",
            source,
        )
    )


def _vector_int64_kind(arg, vector_params, optional_vector_params):
    """Classify a vector parameter, preferring its overload-local type."""
    spelling = "".join(arg.type.spelling.split())
    spelling = spelling.removeprefix("const").removesuffix("&")

    if spelling == "std::optional<std::vector<int64_t>>":
        return "optional"

    if spelling == "std::vector<int64_t>":
        return "vector"

    if spelling != "int":
        return None

    if arg.spelling in optional_vector_params:
        return "optional"

    if arg.spelling in vector_params:
        return "vector"

    return None


def _find_tensor_params(op_name):
    source = _find_base_header(op_name).read_text()

    params = set()
    params.update(_find_optional_vector_tensor_params(op_name))
    params.update(re.findall(r"(?:^|[,(]\s*)(?:const\s+)?Tensor\s+(\w+)", source))
    params.update(_find_optional_tensor_params(op_name))
    params.update(_find_vector_tensor_params(op_name))

    return params


def _find_params_with_defaults(op_name):
    """Return `{param_name: default_literal}` for scalar params with defaults."""
    source = _find_base_header(op_name).read_text()
    mapping = {}

    for name, default in re.findall(
        r"\b(?:bool|int(?:64_t|32_t|8_t|16_t)?|std::size_t|std::uint\w+_t|"
        r"float|double)\s+(\w+)\s*=\s*([^,\)]+?)\s*(?:,|\))",
        source,
    ):
        mapping[name] = default.strip()

    return mapping


def _is_data_type_spelling(spelling):
    spelling = spelling.replace("const ", "").replace("&", "").strip()

    return spelling.rsplit("::", maxsplit=1)[-1] == "DataType"


def _generate_pybind11(operator):
    optional_tensor_params = _find_optional_tensor_params(operator.name)
    optional_non_tensor_params = _find_optional_non_tensor_params(operator.name)
    vector_tensor_params = _find_vector_tensor_params(operator.name)
    optional_vector_tensor_params = _find_optional_vector_tensor_params(operator.name)
    vector_int64_params = _find_vector_int64_params(operator.name)
    optional_vector_int64_params = _find_optional_vector_int64_params(operator.name)
    params_with_defaults = _find_params_with_defaults(operator.name)

    def _is_optional_vector_tensor(arg):
        return (
            _vector_tensor_kind(
                arg, vector_tensor_params, optional_vector_tensor_params
            )
            == "optional"
        )

    def _is_optional_tensor(arg):
        spelling = arg.type.spelling

        if _is_known_non_tensor_type(spelling):
            return False

        if _is_optional_vector_tensor(arg):
            return False

        if "std::optional" in spelling:
            return "Tensor" in spelling

        if arg.spelling in optional_non_tensor_params:
            return False

        return arg.spelling in optional_tensor_params

    def _is_optional(arg):
        spelling = "".join(arg.type.spelling.split()).removeprefix("const")
        return spelling.startswith("std::optional<")

    def _is_optional_vector_int64(arg):
        return (
            _vector_int64_kind(arg, vector_int64_params, optional_vector_int64_params)
            == "optional"
        )

    def _is_vector_tensor(arg):
        return (
            _vector_tensor_kind(
                arg, vector_tensor_params, optional_vector_tensor_params
            )
            == "vector"
        )

    def _is_vector_int64(arg):
        return (
            _vector_int64_kind(arg, vector_int64_params, optional_vector_int64_params)
            == "vector"
        )

    def _is_data_type(arg):
        return _is_data_type_spelling(arg.type.spelling)

    def _generate_params(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_vector_tensor(arg):
                parts.append(f"std::vector<py::object> {arg.spelling}")
            elif _is_optional_tensor(arg):
                parts.append(f"std::optional<py::object> {arg.spelling}")
            elif _is_vector_tensor(arg):
                parts.append(f"std::vector<py::object> {arg.spelling}")
            elif _is_optional_vector_int64(arg):
                parts.append(
                    f"const std::optional<std::vector<int64_t>> {arg.spelling}"
                )
            elif _is_vector_int64(arg):
                parts.append(f"const std::vector<int64_t> {arg.spelling}")
            elif _is_data_type(arg):
                parts.append(f"py::object {arg.spelling}")
            else:
                param = arg.type.spelling.replace("const Tensor", "py::object").replace(
                    "Tensor", "py::object"
                )
                parts.append(f"{param} {arg.spelling}")

        return ", ".join(parts)

    def _generate_arguments(
        node, preconverted_tensor_arg=None, preconverted_tensor_name=None
    ):
        args = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if (
                preconverted_tensor_arg is not None
                and arg.spelling == preconverted_tensor_arg.spelling
            ):
                args.append(preconverted_tensor_name)
            elif _is_optional_vector_tensor(arg):
                args.append(f"VectorOptionalTensorFromPybind11Handle({arg.spelling})")
            elif _is_optional_tensor(arg):
                args.append(f"OptionalTensorFromPybind11Handle({arg.spelling})")
            elif _is_vector_tensor(arg):
                args.append(f"VectorTensorFromPybind11Handle({arg.spelling})")
            elif "Tensor" in arg.type.spelling:
                args.append(f"TensorFromPybind11Handle({arg.spelling})")
            elif _is_data_type(arg):
                args.append(f"DataTypeFromPybind11Handle({arg.spelling})")
            else:
                args.append(arg.spelling)

        return ", ".join(args)

    op_name = operator.name
    op_type = _op_cpp_type(op_name)
    symbol_name = _op_symbol_name(op_name)

    def _first_tensor_arg(node):
        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue
            if _is_optional_tensor(arg) or _is_optional_vector_tensor(arg):
                continue
            if _is_vector_tensor(arg):
                return arg
            if "Tensor" in arg.type.spelling:
                return arg
        return None

    def _unique_local_name(node, base):
        argument_names = {arg.spelling for arg in node.get_arguments()}
        name = base
        while name in argument_names:
            name += "_"
        return name

    def _tensor_conversion_expr(arg):
        if _is_vector_tensor(arg):
            return f"VectorTensorFromPybind11Handle({arg.spelling})"
        return f"TensorFromPybind11Handle({arg.spelling})"

    def _default_impl_index_expr(node, converted_first_tensor_name=None):
        first_tensor_arg = _first_tensor_arg(node)
        if first_tensor_arg is None:
            return "0"

        if converted_first_tensor_name is not None:
            first_tensor = converted_first_tensor_name
            if _is_vector_tensor(first_tensor_arg):
                first_tensor += ".at(0)"
            device_type = f"{first_tensor}.device().type()"
        else:
            first_tensor = first_tensor_arg.spelling
            if _is_vector_tensor(first_tensor_arg):
                first_tensor += ".at(0)"
            device_type = f"DeviceFromPybind11Handle({first_tensor}).type()"

        return f"DefaultImplementationIndexFor{symbol_name}({device_type})"

    def _generate_init(constructor):
        constructor_params = _generate_params(constructor)
        default_impl_index = _default_impl_index_expr(constructor)

        return f"""      .def(py::init([]({constructor_params}) {{
        Config config;
        config.set_implementation_index({default_impl_index});
        return std::unique_ptr<Self>{{static_cast<Self*>(generated_dispatch::Make{symbol_name}(config, {_generate_arguments(constructor)}).release())}};
      }}))"""

    def _generate_py_args(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional(arg):
                parts.append(f'py::arg("{arg.spelling}") = py::none()')
            elif arg.spelling in params_with_defaults:
                parts.append(
                    f'py::arg("{arg.spelling}") = {params_with_defaults[arg.spelling]}'
                )
            else:
                parts.append(f'py::arg("{arg.spelling}")')

        return ", ".join(parts)

    def _generate_call(op_name, call, method=True, supports_triton_config=False):
        call_params = _generate_params(call)
        call_args = _generate_arguments(call)

        if not method:
            first_tensor_arg = _first_tensor_arg(call)
            converted_first_tensor_name = None
            first_tensor_conversion = ""
            if first_tensor_arg is not None:
                converted_first_tensor_name = _unique_local_name(
                    call, "converted_first_tensor"
                )
                first_tensor_conversion = (
                    f"    auto {converted_first_tensor_name}"
                    f"{{{_tensor_conversion_expr(first_tensor_arg)}}};\n"
                )
                call_args = _generate_arguments(
                    call, first_tensor_arg, converted_first_tensor_name
                )
            extra_params = ""
            extra_config_init = ""
            extra_pybind = ""
            if supports_triton_config:
                extra_params = ", std::optional<py::dict> config_dict"
                extra_config_init = (
                    "    std::unique_ptr<Config> triton_config_ptr;\n"
                    "    if (config_dict.has_value()) {\n"
                    "      triton_config_ptr = "
                    "triton::jit::ConfigFromPyDict(*config_dict);\n"
                    "      if (!config.needs_implementation_resolution()) {\n"
                    "        triton_config_ptr->set_implementation_index(\n"
                    "            config.implementation_index());\n"
                    "      }\n"
                    "    }\n"
                )
                extra_pybind = ', py::arg("config") = py::none()'

            params = (
                f"{call_params}, std::uintptr_t stream, "
                f"std::optional<std::size_t> implementation_index{extra_params}"
                if call_params
                else f"std::uintptr_t stream, std::optional<std::size_t> implementation_index{extra_params}"
            )
            py_args = _generate_py_args(call)
            py_args_str = f"{py_args}, " if py_args else ""
            default_impl_index = _default_impl_index_expr(
                call, converted_first_tensor_name
            )

            if supports_triton_config:
                dispatch = (
                    "    if (triton_config_ptr) {\n"
                    f"      auto op = generated_dispatch::Make{symbol_name}(*triton_config_ptr, {call_args});\n"
                    f"      return (*op)(handle, {call_args});\n"
                    "    }\n"
                    f"    return generated_dispatch::Call{symbol_name}(handle, config, {call_args});"
                )
            else:
                dispatch = f"    return generated_dispatch::Call{symbol_name}(handle, config, {call_args});"

            return (
                f'  m.def("{op_name}", []({params}) {{\n'
                f"    [[maybe_unused]] HostRangeScope host_range_binding_body{{\n"
                f"        HostRangeLayer::kBindingBody}};\n"
                f"    Handle handle;\n"
                f"    if (stream) {{\n"
                f"      handle.set_stream(reinterpret_cast<void*>(stream));\n"
                f"    }}\n"
                f"{first_tensor_conversion}"
                f"    Config config;\n"
                f"    if (implementation_index.has_value()) {{\n"
                f"      config.set_implementation_index(*implementation_index);\n"
                f"    }} else if (!TuningManager::Instance().IsEnabled()) {{\n"
                f"      config.set_implementation_index(\n"
                f"          {default_impl_index});\n"
                f"    }}\n"
                f"{extra_config_init}"
                f"{dispatch}\n"
                f'  }}, {py_args_str}py::kw_only(), py::arg("stream") = 0, py::arg("implementation_index") = py::none(){extra_pybind});'
            )

        # The first lambda parameter is conventionally named `self`, but
        # ATen schemas often have a parameter literally called `self`
        # (e.g. `pow.Tensor_Scalar_out(Scalar self, Tensor exponent)`),
        # so rename to `op` to avoid the collision in the generated code.

        return f"""      .def("__call__", [](const Self& op, {call_params}) {{
        return generated_dispatch::Invoke{symbol_name}(op, {call_args});
      }})"""

    def _overload_order_key(node):
        """Sort key that places more-specific overloads first.

        Tensor parameters are exposed to pybind as `py::object`, which
        accepts any Python value and only fails inside
        `TensorFromPybind11Handle`.  When a class has both Tensor and
        scalar overloads, pybind's overload-resolver tries them in
        registration order and stops at the first that does not raise,
        so the scalar overload must be registered first; otherwise the
        permissive Tensor signature swallows scalar calls and aborts at
        runtime.
        """
        object_like = 0
        total = 0

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            total += 1

            if (
                _is_optional_tensor(arg)
                or _is_vector_tensor(arg)
                or "Tensor" in arg.type.spelling
            ):
                object_like += 1

        return (object_like, -total)

    constructors = sorted(operator.constructors, key=_overload_order_key)
    operator_calls = sorted(operator.calls, key=_overload_order_key)

    inits = "\n".join(_generate_init(constructor) for constructor in constructors)
    calls = "\n".join(_generate_call(operator.name, call) for call in operator_calls)

    supports_triton = operator.has_backend("triton")
    constructor_signatures = {
        _parameter_type_signature(constructor) for constructor in constructors
    }
    configurable_calls = [
        supports_triton and _parameter_type_signature(call) in constructor_signatures
        for call in operator_calls
    ]
    callers = "\n".join(
        _generate_call(
            operator.name,
            call,
            method=False,
            supports_triton_config=supports_config,
        )
        for call, supports_config in zip(operator_calls, configurable_calls)
    )
    triton_config_include = (
        '\n#include "triton/jit/pybind11_config.h"' if any(configurable_calls) else ""
    )

    return f"""#ifndef INFINI_OPS_BINDINGS_{op_name.upper()}_H_
#define INFINI_OPS_BINDINGS_{op_name.upper()}_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

#include "base/{op_name}.h"
#include "config.h"
#include "generated/bindings/generated_dispatch.h"
#include "handle.h"
#include "host_range_profiler.h"
#include "pybind11_utils.h"
#include "tuning.h"{triton_config_include}

namespace py = pybind11;

namespace infini::ops {{

std::size_t DefaultImplementationIndexFor{symbol_name}(Device::Type dev_type) {{
  return {op_type}::DefaultConfig(dev_type).implementation_index();
}}

void Bind{symbol_name}(py::module& m) {{
  using Self = {op_type};

  py::class_<Self>(m, "{symbol_name}")
{inits}
{calls}
      .def_static("active_implementation_indices", [](const std::string& device) {{
        auto dev_type = TryDeviceTypeFromString<Self>(device);
        if (!dev_type.has_value()) {{
          return std::vector<std::size_t>{{}};
        }}
        return generated_dispatch::ActiveImplementationIndicesFor{symbol_name}(*dev_type);
      }})
      .def_static("clear_cache", &generated_dispatch::ClearCacheFor{symbol_name});

{callers}
}}

}}  // namespace infini::ops

#endif
"""


def _generate_legacy_c(operator, paths):
    op_type = _op_cpp_type(operator.name)
    symbol_name = _op_symbol_name(operator.name)

    def _generate_source(operator):
        impl_includes = "\n".join(
            f'#include "{_to_include_path(path)}"' for path in paths
        )

        return f"""#include "../../handle.h"
#include "../../tensor.h"
#include "infiniop/ops/{operator.name.lower()}.h"
{impl_includes}

static infini::ops::DataType DataTypeFromInfiniDType(
    const infiniDtype_t& dtype) {{
  static constexpr infini::ops::ConstexprMap<infiniDtype_t,
                                             infini::ops::DataType, 12>
      kInfiniDTypeToDataType{{
          {{{{{{INFINI_DTYPE_I8, infini::ops::DataType::kInt8}},
            {{INFINI_DTYPE_I16, infini::ops::DataType::kInt16}},
            {{INFINI_DTYPE_I32, infini::ops::DataType::kInt32}},
            {{INFINI_DTYPE_I64, infini::ops::DataType::kInt64}},
            {{INFINI_DTYPE_U8, infini::ops::DataType::kUInt8}},
            {{INFINI_DTYPE_U16, infini::ops::DataType::kUInt16}},
            {{INFINI_DTYPE_U32, infini::ops::DataType::kUInt32}},
            {{INFINI_DTYPE_U64, infini::ops::DataType::kUInt64}},
            {{INFINI_DTYPE_F16, infini::ops::DataType::kFloat16}},
            {{INFINI_DTYPE_BF16, infini::ops::DataType::kBFloat16}},
            {{INFINI_DTYPE_F32, infini::ops::DataType::kFloat32}},
            {{INFINI_DTYPE_F64, infini::ops::DataType::kFloat64}}}}}}}};

  return kInfiniDTypeToDataType.at(dtype);
}}

static infini::ops::Device::Type DeviceTypeFromInfiniDevice(
    const infiniDevice_t& device) {{
  static constexpr infini::ops::ConstexprMap<
      infiniDevice_t, infini::ops::Device::Type,
      static_cast<std::size_t>(INFINI_DEVICE_TYPE_COUNT)>
      kInfiniDeviceToDeviceType{{
          {{{{{{INFINI_DEVICE_CPU, infini::ops::Device::Type::kCpu}},
            {{INFINI_DEVICE_NVIDIA, infini::ops::Device::Type::kNvidia}},
            {{INFINI_DEVICE_CAMBRICON, infini::ops::Device::Type::kCambricon}},
            {{INFINI_DEVICE_ASCEND, infini::ops::Device::Type::kAscend}},
            {{INFINI_DEVICE_METAX, infini::ops::Device::Type::kMetax}},
            {{INFINI_DEVICE_MOORE, infini::ops::Device::Type::kMoore}},
            {{INFINI_DEVICE_ILUVATAR, infini::ops::Device::Type::kIluvatar}},
            {{INFINI_DEVICE_HYGON, infini::ops::Device::Type::kHygon}},
            {{INFINI_DEVICE_ALI, infini::ops::Device::Type::kThead}}}}}}}};

  return kInfiniDeviceToDeviceType.at(device);
}}

__C {_generate_create_func_def(operator)}

__C {_generate_get_workspace_size_func_def(operator)}

__C {_generate_call_func_def(operator)}

__C {_generate_destroy_func_def(operator)}
"""

    def _generate_header(operator):
        return f"""#ifndef __INFINIOP_{operator.name.upper()}_API_H__
#define __INFINIOP_{operator.name.upper()}_API_H__

#include "base/{operator.name.lower()}.h"

typedef struct infini::ops::Operator<{op_type}> *infiniop{symbol_name}Descriptor_t;

__C __export {_generate_create_func_decl(operator)};

__C __export {_generate_get_workspace_size_func_decl(operator)};

__C __export {_generate_call_func_decl(operator)};

__C __export {_generate_destroy_func_decl(operator)};

#endif
"""

    def _generate_create_func_def(operator):
        constructor = operator.constructors[-1]

        return f"""{_generate_create_func_decl(operator)} {{
    *desc_ptr = infini::ops::Operator<{op_type}>::Make({_generate_arguments(constructor)}).release();

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_get_workspace_size_func_def(operator):
        return f"""{_generate_get_workspace_size_func_decl(operator)} {{
    *size = 0;  // desc->workspace_size();

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_call_func_def(operator):
        call = operator.calls[-1]

        return f"""{_generate_call_func_decl(operator)} {{
    (*desc)(stream, {_generate_arguments(call, is_data=True)});

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_destroy_func_def(operator):
        return f"""{_generate_destroy_func_decl(operator)} {{
    delete desc;

    return INFINI_STATUS_SUCCESS;
}}"""

    def _generate_create_func_decl(operator):
        constructor = operator.constructors[-1]
        params = _generate_params(constructor)

        return f"infiniStatus_t infiniopCreate{symbol_name}Descriptor(infiniopHandle_t handle, infiniop{symbol_name}Descriptor_t *desc_ptr, {params})"

    def _generate_get_workspace_size_func_decl(operator):
        return f"infiniStatus_t infiniopGet{symbol_name}WorkspaceSize(infiniop{symbol_name}Descriptor_t desc, size_t *size)"

    def _generate_call_func_decl(operator):
        call = operator.calls[-1]
        params = _generate_params(call, call=True)
        params = params.replace("void * stream, ", "")

        return f"infiniStatus_t infiniop{symbol_name}(infiniop{symbol_name}Descriptor_t desc, void *workspace, size_t workspace_size, {params}, void *stream)"

    def _generate_destroy_func_decl(operator):
        return f"infiniStatus_t infiniopDestroy{symbol_name}Descriptor(infiniop{symbol_name}Descriptor_t desc)"

    def _generate_params(node, call=False):
        arguments = tuple(node.get_arguments())

        arguments = (arguments[-1], *arguments[:-1])

        def _unwrap_std_optional(spelling):
            prefix = "std::optional<"

            if not spelling.startswith(prefix):
                return spelling

            inner = spelling[len(prefix) :]

            if inner.endswith(" >"):
                return inner[:-2] + ">"

            if inner.endswith(">"):
                return inner[:-1]

            return inner

        def _handle_tensor(spelling):
            if call:
                return spelling.replace("Tensor", "void *")

            return spelling.replace("Tensor", "infiniopTensorDescriptor_t")

        def _handle_std_optional(spelling):
            return _unwrap_std_optional(spelling)

        def _handle_data_type(spelling):
            if not _is_data_type_spelling(spelling):
                return spelling

            prefix = "const " if spelling.strip().startswith("const ") else ""

            return f"{prefix}infiniDtype_t"

        return ", ".join(
            f"{_handle_data_type(_handle_std_optional(_handle_tensor(arg.type.spelling)))} {arg.spelling}"
            for arg in arguments
        )

    def _generate_arguments(node, is_data=False):
        return ", ".join(
            f"DataTypeFromInfiniDType({arg.spelling})"
            if _is_data_type_spelling(arg.type.spelling)
            else (
                _generate_tensor_caster(arg.spelling, is_data=is_data)
                if "Tensor" in arg.type.spelling
                else arg.spelling
            )
            for arg in node.get_arguments()
            if arg.spelling != "handle" and arg.spelling != "stream"
        )

    def _generate_tensor_caster(name, is_data=False):
        if is_data:
            return f"infini::ops::Tensor(const_cast<void *>({name}), infini::ops::Tensor::Shape{{}})"

        return f"infini::ops::Tensor{{nullptr, {name}->shape(), DataTypeFromInfiniDType({name}->dtype()), infini::ops::Device{{DeviceTypeFromInfiniDevice(handle->device), handle->device_id}}, {name}->strides()}}"

    return _generate_source(operator), _generate_header(operator)


def _generate_generated_dispatch_entries(operator):
    optional_tensor_params = _find_optional_tensor_params(operator.name)
    optional_non_tensor_params = _find_optional_non_tensor_params(operator.name)
    tensor_params = _find_tensor_params(operator.name)
    vector_tensor_params = _find_vector_tensor_params(operator.name)
    optional_vector_tensor_params = _find_optional_vector_tensor_params(operator.name)
    vector_int64_params = _find_vector_int64_params(operator.name)
    optional_vector_int64_params = _find_optional_vector_int64_params(operator.name)

    def _is_optional_vector_tensor(arg):
        return (
            _vector_tensor_kind(
                arg, vector_tensor_params, optional_vector_tensor_params
            )
            == "optional"
        )

    def _is_optional_tensor(arg):
        spelling = arg.type.spelling

        if _is_known_non_tensor_type(spelling):
            return False

        if _is_optional_vector_tensor(arg):
            return False

        if "std::optional" in spelling:
            return "Tensor" in spelling

        if arg.spelling in optional_non_tensor_params:
            return False

        return arg.spelling in optional_tensor_params

    def _is_optional_vector_int64(arg):
        return (
            _vector_int64_kind(arg, vector_int64_params, optional_vector_int64_params)
            == "optional"
        )

    def _is_vector_tensor(arg):
        return (
            _vector_tensor_kind(
                arg, vector_tensor_params, optional_vector_tensor_params
            )
            == "vector"
        )

    def _is_vector_int64(arg):
        return (
            _vector_int64_kind(arg, vector_int64_params, optional_vector_int64_params)
            == "vector"
        )

    def _is_tensor(arg):
        if arg.spelling in optional_non_tensor_params:
            return False

        if "Tensor" in arg.type.spelling or "TensorView" in arg.type.spelling:
            return True

        if _is_known_non_tensor_type(arg.type.spelling):
            return False

        return arg.spelling in tensor_params

    def _generate_params(node):
        parts = []

        for arg in node.get_arguments():
            if arg.spelling == "stream":
                continue

            if _is_optional_vector_tensor(arg):
                parts.append(f"std::vector<std::optional<Tensor>> {arg.spelling}")
            elif _is_optional_tensor(arg):
                parts.append(f"std::optional<Tensor> {arg.spelling}")
            elif _is_vector_tensor(arg):
                parts.append(f"std::vector<Tensor> {arg.spelling}")
            elif _is_optional_vector_int64(arg):
                parts.append(f"std::optional<std::vector<int64_t>> {arg.spelling}")
            elif _is_vector_int64(arg):
                parts.append(f"std::vector<int64_t> {arg.spelling}")
            elif _is_tensor(arg):
                parts.append(f"Tensor {arg.spelling}")
            else:
                parts.append(f"{arg.type.spelling} {arg.spelling}")

        return ", ".join(parts)

    def _generate_arguments(node):
        return ", ".join(
            arg.spelling for arg in node.get_arguments() if arg.spelling != "stream"
        )

    def _append_optional_args(prefix, args):
        if args:
            return f"{prefix}, {args}"

        return prefix

    def _append_optional_params(prefix, params):
        if params:
            return f"{prefix}, {params}"

        return prefix

    symbol_name = _op_symbol_name(operator.name)
    op_type = _op_cpp_type(operator.name)
    declarations = [
        f"std::vector<std::size_t> ActiveImplementationIndicesFor"
        f"{symbol_name}(Device::Type dev_type);"
    ]
    definitions = [
        f"""std::vector<std::size_t> ActiveImplementationIndicesFor{symbol_name}(Device::Type dev_type) {{
  return Operator<{op_type}>::active_implementation_indices(dev_type);
}}"""
    ]

    declarations.append(f"void ClearCacheFor{symbol_name}();")
    definitions.append(
        f"""#if !defined(INFINI_OPS_USE_OPERATOR_CALL_INSTANTIATIONS)
void ClearCacheFor{symbol_name}() {{
  Operator<{op_type}>::clear_cache();
}}
#endif"""
    )

    emitted_make_params = set()

    for node in operator.constructors:
        params = _generate_params(node)
        constructor_args = [
            arg.spelling for arg in node.get_arguments() if arg.spelling != "stream"
        ]
        # Operator::Make forwards its trailing arguments. Named by-value
        # dispatch parameters are mutable lvalues, while the generated public
        # instantiations use const T& for every trailing argument. Canonicalize
        # this call so Python bindings request the exported specialization.
        args = ", ".join(
            [constructor_args[0]]
            + [f"std::as_const({name})" for name in constructor_args[1:]]
        )
        make_params = _append_optional_params("const Config& config", params)

        if make_params in emitted_make_params:
            continue

        emitted_make_params.add(make_params)
        declarations.append(
            f"std::unique_ptr<Operator<{op_type}>> Make{symbol_name}({make_params});"
        )
        definitions.append(
            f"""std::unique_ptr<Operator<{op_type}>> Make{symbol_name}({make_params}) {{
  return Operator<{op_type}>::Make({_append_optional_args("config", args)});
}}"""
        )

    emitted_call_params = set()

    for call in operator.calls:
        params = _generate_params(call)
        args = _generate_arguments(call)

        if params in emitted_call_params:
            continue

        emitted_call_params.add(params)
        declarations.append(
            f"void Invoke{symbol_name}(const "
            f"{_append_optional_params(f'{op_type}& op', params)});"
        )
        definitions.append(
            f"""void Invoke{symbol_name}(const {_append_optional_params(f"{op_type}& op", params)}) {{
  return static_cast<const Operator<{op_type}>&>(op)({args});
}}"""
        )

        declarations.append(
            f"void Call{symbol_name}(const Handle& handle, "
            f"{_append_optional_params('const Config& config', params)});"
        )
        definitions.append(
            f"""void Call{symbol_name}(const Handle& handle, {_append_optional_params("const Config& config", params)}) {{
  [[maybe_unused]] HostRangeScope host_range_dispatch_call{{
      HostRangeLayer::kDispatchCall}};
  return Operator<{op_type}>::Call({_append_optional_args("handle, config", args)});
}}"""
        )

    return declarations, definitions


def _generate_generated_dispatch_header(op_names, devices, declarations):
    header_base_includes = "\n".join(
        f'#include "base/{op_name}.h"' for op_name in op_names
    )
    header_device_includes = "\n".join(
        f'#include "{path}"' for path in _device_marker_headers(devices)
    )

    return f"""#ifndef INFINI_OPS_GENERATED_BINDINGS_GENERATED_DISPATCH_H_
#define INFINI_OPS_GENERATED_BINDINGS_GENERATED_DISPATCH_H_

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include "config.h"
#include "device.h"
#include "handle.h"
#include "operator.h"

{header_device_includes}

{header_base_includes}

namespace infini::ops::generated_dispatch {{

{chr(10).join(declarations)}

}}  // namespace infini::ops::generated_dispatch

#endif
"""


def _generate_generated_dispatch_source(impl_paths, definitions):
    impl_includes = "\n".join(f'#include "{impl_path}"' for impl_path in impl_paths)

    return f"""#include "generated_dispatch.h"
#include "infini/operator_call_instantiations.h"
#include "host_range_profiler.h"

#include <utility>

// clang-format off
{impl_includes}
// clang-format on

namespace infini::ops::generated_dispatch {{

{chr(10).join(definitions)}

}}  // namespace infini::ops::generated_dispatch
"""


def _generate_cache_clear_dispatch_source(op_names):
    base_includes = "\n".join(f'#include "base/{name}.h"' for name in op_names)
    definitions = "\n\n".join(
        f"""void ClearCacheFor{_op_symbol_name(name)}() {{
  Operator<{_op_cpp_type(name)}>::clear_cache();
}}"""
        for name in op_names
    )

    return f"""#include "operator.h"

{base_includes}

namespace infini::ops::generated_dispatch {{

{definitions}

}}  // namespace infini::ops::generated_dispatch
"""


def _strip_top_level_const(type_spelling):
    type_spelling = " ".join(type_spelling.split())

    while type_spelling.startswith("const "):
        type_spelling = type_spelling[len("const ") :]

    return type_spelling


def _is_known_non_tensor_type(type_spelling):
    type_spelling = _strip_top_level_const(type_spelling)

    if "Tensor" in type_spelling or "TensorView" in type_spelling:
        return False

    if type_spelling in {
        "bool",
        "double",
        "float",
        "int64_t",
        "std::size_t",
        "std::string",
    }:
        return True

    return type_spelling.startswith("std::optional<")


def _generate_operator_call_instantiation_entries(operator):
    optional_tensor_params = _find_optional_tensor_params(operator.name)
    optional_non_tensor_params = _find_optional_non_tensor_params(operator.name)
    tensor_params = _find_tensor_params(operator.name)
    optional_vector_tensor_params = _find_optional_vector_tensor_params(operator.name)
    vector_tensor_params = _find_vector_tensor_params(operator.name)
    vector_int64_params = _find_vector_int64_params(operator.name)
    optional_vector_int64_params = _find_optional_vector_int64_params(operator.name)

    def _is_optional_vector_tensor(arg):
        return (
            _vector_tensor_kind(
                arg, vector_tensor_params, optional_vector_tensor_params
            )
            == "optional"
        )

    def _is_optional_tensor(arg):
        if _is_optional_vector_tensor(arg):
            return False

        spelling = arg.type.spelling

        if _is_known_non_tensor_type(spelling):
            return False

        if "std::optional" in spelling:
            return "Tensor" in spelling or "TensorView" in spelling

        if arg.spelling in optional_non_tensor_params:
            return False

        if arg.spelling in optional_tensor_params:
            return True

        return False

    def _is_optional_vector_int64(arg):
        return (
            _vector_int64_kind(arg, vector_int64_params, optional_vector_int64_params)
            == "optional"
        )

    def _is_vector_tensor(arg):
        return (
            _vector_tensor_kind(
                arg, vector_tensor_params, optional_vector_tensor_params
            )
            == "vector"
        )

    def _is_vector_int64(arg):
        return (
            _vector_int64_kind(arg, vector_int64_params, optional_vector_int64_params)
            == "vector"
        )

    def _is_tensor(arg):
        if arg.spelling in optional_non_tensor_params:
            return False

        if "Tensor" in arg.type.spelling or "TensorView" in arg.type.spelling:
            return True

        if _is_known_non_tensor_type(arg.type.spelling):
            return False

        return arg.spelling in tensor_params

    def _normalized_type(arg):
        if _is_optional_vector_tensor(arg):
            return "std::vector<std::optional<Tensor>>"

        if _is_optional_tensor(arg):
            return "std::optional<Tensor>"

        if _is_vector_tensor(arg):
            return "std::vector<Tensor>"

        if _is_optional_vector_int64(arg):
            return "std::optional<std::vector<int64_t>>"

        if _is_vector_int64(arg):
            return "std::vector<int64_t>"

        if _is_tensor(arg):
            return "Tensor"

        return _strip_top_level_const(arg.type.spelling)

    def _generate_template_arguments(node):
        return ", ".join(
            _normalized_type(arg)
            for arg in node.get_arguments()
            if arg.spelling != "stream"
        )

    def _generate_parameters(node):
        return ", ".join(
            f"const {_normalized_type(arg)}& {arg.spelling}"
            for arg in node.get_arguments()
            if arg.spelling != "stream"
        )

    def _append_optional_params(prefix, params):
        if params:
            return f"{prefix}, {params}"

        return prefix

    op_type = _op_cpp_type(operator.name)
    declarations = []
    definitions = []
    factory_declarations = []
    factory_definitions = []

    seen_entries = set()

    def _append_unique(declaration, definition):
        if definition in seen_entries:
            return

        seen_entries.add(definition)
        declarations.append(declaration)
        definitions.append(definition)

    _append_unique(
        f"extern template std::vector<std::size_t> "
        f"Operator<{op_type}>::active_implementation_indices(Device::Type);",
        f"template std::vector<std::size_t> "
        f"Operator<{op_type}>::active_implementation_indices(Device::Type);",
    )

    _append_unique(
        f"extern template Config Operator<{op_type}>::DefaultConfig(Device::Type);",
        f"template Config Operator<{op_type}>::DefaultConfig(Device::Type);",
    )

    _append_unique(
        f"extern template std::size_t "
        f"Operator<{op_type}>::DefaultImplementationIndex(Device::Type);",
        f"template std::size_t "
        f"Operator<{op_type}>::DefaultImplementationIndex(Device::Type);",
    )

    for call in operator.calls:
        template_arguments = _generate_template_arguments(call)
        params = _generate_parameters(call)
        function_params = _append_optional_params(
            "const Handle& handle, const Config& config", params
        )
        args = [arg for arg in call.get_arguments() if arg.spelling != "stream"]
        first_arg = args[0]
        rest_args = args[1:]
        instantiation = (
            f"Operator<{op_type}>::Call<{template_arguments}>({function_params})"
        )

        make_template_arguments = ", ".join(
            f"const {_normalized_type(arg)}&" for arg in rest_args
        )
        make_params = ", ".join(
            f"const {_normalized_type(arg)}& {arg.spelling}" for arg in rest_args
        )
        make_function_params = _append_optional_params(
            f"const Config& config, const {_normalized_type(first_arg)} {first_arg.spelling}",
            make_params,
        )
        make_instantiation = f"Operator<{op_type}>::Make<{make_template_arguments}>({make_function_params})"
        can_make_template_arguments = ", ".join(
            _normalized_type(arg) for arg in rest_args
        )
        can_make_instantiation = (
            f"Operator<{op_type}>::CanMake<{can_make_template_arguments}>"
            f"({make_function_params})"
        )

        operator_instantiation = (
            f"Operator<{op_type}>::operator()<{template_arguments}>({params}) const"
        )

        _append_unique(
            f"extern template void {instantiation};",
            f"template void {instantiation};",
        )
        _append_unique(
            f"extern template std::unique_ptr<Operator<{op_type}>> {make_instantiation};",
            f"template std::unique_ptr<Operator<{op_type}>> {make_instantiation};",
        )
        _append_unique(
            f"extern template bool {can_make_instantiation};",
            f"template bool {can_make_instantiation};",
        )
        _append_unique(
            f"extern template void {operator_instantiation};",
            f"template void {operator_instantiation};",
        )

        factory_declarations.append(f"  static void Call({function_params});")
        factory_definitions.append(
            f"void detail::OperatorFactory<{op_type}>::Call({function_params}) {{\n"
            f"  return Operator<{op_type}>::Call(handle, config, "
            f"{', '.join(arg.spelling for arg in args)});\n}}"
        )

    for constructor in operator.constructors or operator.calls:
        args = [arg for arg in constructor.get_arguments() if arg.spelling != "stream"]
        params = ", ".join(
            f"const {_normalized_type(arg)}& {arg.spelling}" for arg in args
        )
        formal_params = _append_optional_params("const Config& config", params)
        arguments = ", ".join(arg.spelling for arg in args)
        factory_declarations.extend(
            [
                f"  static std::unique_ptr<Operator<{op_type}>> Make({formal_params});",
                f"  static bool CanMake({formal_params});",
            ]
        )
        factory_definitions.extend(
            [
                f"std::unique_ptr<Operator<{op_type}>> detail::OperatorFactory<{op_type}>::Make({formal_params}) {{\n"
                f"  return Operator<{op_type}>::Make(config, {arguments});\n}}",
                f"bool detail::OperatorFactory<{op_type}>::CanMake({formal_params}) {{\n"
                f"  return Operator<{op_type}>::CanMake(config, {arguments});\n}}",
            ]
        )

    declarations.append(
        f"template <>\nstruct detail::OperatorFactory<{op_type}> {{\n"
        + "\n".join(dict.fromkeys(factory_declarations))
        + "\n};"
    )
    definitions.extend(dict.fromkeys(factory_definitions))

    return declarations, definitions


def _generate_operator_call_instantiation_header(op_names, declarations):
    header_base_includes = "\n".join(
        f'#include "base/{op_name}.h"' for op_name in op_names
    )

    return f"""#ifndef INFINI_OPS_OPERATOR_CALL_INSTANTIATIONS_H_
#define INFINI_OPS_OPERATOR_CALL_INSTANTIATIONS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "config.h"
#include "handle.h"
#include "operator.h"

{header_base_includes}

namespace infini::ops {{

{chr(10).join(declarations)}

}}  // namespace infini::ops

#endif
"""


def _generate_operator_call_instantiation_source(devices, impl_paths, definitions):
    device_includes = "\n".join(
        f'#include "{path}"' for path in _device_marker_headers(devices)
    )
    impl_includes = "\n".join(
        f'#include "{_to_include_path(impl_path)}"' for impl_path in impl_paths
    )

    return f"""#include "infini/operator_call_instantiations.h"

// clang-format off
{device_includes}
{impl_includes}
// clang-format on

namespace infini::ops {{

{chr(10).join(definitions)}

}}  // namespace infini::ops
"""


def _device_marker_headers(devices):
    paths = {
        "cpu": "infini/rt/cpu/device_.h",
        "nvidia": "infini/rt/nvidia/device_.h",
        "cambricon": "infini/rt/cambricon/device_.h",
        "ascend": "infini/rt/ascend/device_.h",
        "metax": "infini/rt/metax/device_.h",
        "mars": "infini/rt/mars/device_.h",
        "moore": "infini/rt/moore/device_.h",
        "iluvatar": "infini/rt/iluvatar/device_.h",
        "hygon": "infini/rt/hygon/device_.h",
    }

    return [paths[device] for device in devices if device in paths]


def _generate_binding_source(op_name):
    return f"""#include "{op_name}.h"
"""


def _snake_to_pascal(snake_str):
    return "".join(word.capitalize() for word in snake_str.split("_"))


def _to_include_path(path):
    text = str(path)

    for prefix in ("src/", "generated/"):
        if text.startswith(prefix):
            return text[len(prefix) :]

    return text


def _matches_scan_dir(impl_path, scan_dirs):
    linked_root = _SRC_DIR / "linked"
    try:
        linked_path = impl_path.relative_to(linked_root)
    except ValueError:
        linked_path = None

    if linked_path is not None:
        if "linked" not in scan_dirs:
            return False

        try:
            ops_index = linked_path.parts.index("ops")
        except ValueError:
            return False

        platform_parts = linked_path.parts[:ops_index]
        if not platform_parts:
            return False

        active_devices = scan_dirs - {"linked", "ninetoothed", "torch"}
        return any(part in active_devices for part in platform_parts)

    return any(part in scan_dirs for part in impl_path.parts)


def _implementation_backend(path):
    for backend in ("triton", "torch", "ninetoothed"):
        if backend in path.parts:
            return backend

    return "native"


def _implementation_from_json(implementation):
    if isinstance(implementation, str):
        path = pathlib.Path(implementation)
        return _Implementation(path, _implementation_backend(path))

    return _Implementation(
        pathlib.Path(implementation["path"]), implementation["backend"]
    )


_OPERATOR_DECL_RE = re.compile(
    r"\bclass\s+Operator<\s*((?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)\b"
)


def _index_impl_headers(impl_roots, scan_dirs):
    """Index implementation headers by base operator class name.

    The previous implementation scanned every implementation header once per
    operator.  With the generated PyTorch backend enabled this becomes hundreds
    of ops times hundreds of headers during CMake configure.  Read each header
    once instead and keep the same insertion order as the old nested loops.
    """
    by_operator = {}

    for impl_root in impl_roots:
        for impl_path in impl_root.rglob("*.h"):
            if not _matches_scan_dir(impl_path, scan_dirs):
                continue

            text = impl_path.read_text()

            for match in _OPERATOR_DECL_RE.finditer(text):
                implementation = _Implementation(
                    impl_path, _implementation_backend(impl_path)
                )
                by_operator.setdefault(match.group(1), []).append(implementation)

    return by_operator


def _normalize_op_allowlist(raw_ops):
    if not raw_ops:
        return []

    op_names = []

    for item in raw_ops:
        op_names.extend(name.strip() for name in item.split(","))

    return [name for name in op_names if name]


def _filter_ops(ops, op_allowlist, *, strict=False):
    if not op_allowlist:
        return ops

    missing = [op_name for op_name in op_allowlist if op_name not in ops]

    if missing:
        message = "operator(s) not available for active devices: " + ", ".join(missing)

        if strict:
            raise ValueError(message)

        print(f"warning: {message}")

    return {op_name: ops[op_name] for op_name in op_allowlist if op_name in ops}


def _select_ops_from_config(ops, config, config_path):
    selected = {}

    for op_name, selection in config.items():
        if selection.get("public_api_only") is True:
            base_header = _BASE_DIR / f"{op_name}.h"
            if (
                base_header.resolve().parent != _BASE_DIR.resolve()
                or not base_header.is_file()
            ):
                raise ValueError(
                    f"{config_path}: public API-only operator {op_name!r} "
                    "has no hand-written base header"
                )
            selected[op_name] = []
            continue
        headers = selection["headers"]

        if headers is not None:
            requested = [_implementation_from_json(header) for header in headers]
            active = {impl.path.resolve() for impl in ops.get(op_name, [])}
            for impl in requested:
                # Built-in headers still require their compiled backend.
                # Preserve explicit external-provider header support.
                if (
                    impl.path.resolve().is_relative_to(_SRC_DIR.resolve())
                    and impl.path.resolve() not in active
                ):
                    raise ValueError(
                        f"{config_path}: provider {impl.path} for {op_name!r} "
                        "is not enabled by this build"
                    )
            selected[op_name] = requested
            continue

        if op_name not in ops:
            raise ValueError(
                f"{config_path}: operator {op_name!r} is not available for "
                "the active devices"
            )

        slots = selection["implementations"]

        if slots is None:
            selected[op_name] = ops[op_name]
            continue

        headers_by_slot = {}

        for implementation in ops[op_name]:
            slot = ops_config.implementation_slot(implementation.path)
            headers_by_slot.setdefault(slot, []).append(implementation)

        missing = [slot for slot in slots if slot not in headers_by_slot]

        if missing:
            formatted = ", ".join(str(slot) for slot in missing)
            raise ValueError(
                f"{config_path}: operator {op_name!r} has no active "
                f"implementation at slot(s) {formatted}"
            )

        selected[op_name] = [
            implementation for slot in slots for implementation in headers_by_slot[slot]
        ]

    return selected


def _get_all_ops(
    devices,
    with_torch=False,
    with_ninetoothed=False,
    with_linked=False,
    with_triton=False,
    with_native_flash=False,
    with_native_w4=False,
):
    scan_dirs = set(devices)

    if with_torch:
        scan_dirs.add("torch")
    if with_ninetoothed:
        scan_dirs.add("ninetoothed")
    if with_linked:
        scan_dirs.add("linked")
    if with_triton:
        scan_dirs.add("triton")

    ops = {}

    base_dirs = [_BASE_DIR]

    # Only pull in the auto-generated torch op bases when the build is
    # actually compiling them (`--with-torch`). Otherwise a stale
    # `generated/` left over from a previous configure (or rsynced into
    # a CI container) would cause `ops.cc` to include base headers for
    # ops that have no compiled implementation, breaking the build.
    if with_torch and _GENERATED_BASE_DIR.exists():
        base_dirs.append(_GENERATED_BASE_DIR)

    impl_roots = [_SRC_DIR]

    if with_torch and (_GENERATION_DIR / "torch").exists():
        impl_roots.append(_GENERATION_DIR)

    impl_headers_by_operator = _index_impl_headers(impl_roots, scan_dirs)
    excluded_headers = set()
    if not with_native_flash:
        excluded_headers.update(
            {
                (
                    _SRC_DIR / "native/cuda/nvidia/ops/flash_attn_with_kvcache/kernel.h"
                ).resolve(),
                (
                    _SRC_DIR / "native/cuda/nvidia/ops/flash_attn_varlen_func/kernel.h"
                ).resolve(),
            }
        )
    if not with_native_w4:
        excluded_headers.add(
            (_SRC_DIR / "native/cuda/nvidia/ops/w4_marlin/kernel.h").resolve()
        )
    if excluded_headers:
        # Optional providers are emitted only with their compiled kernel target.
        for name, headers in impl_headers_by_operator.items():
            impl_headers_by_operator[name] = [
                header
                for header in headers
                if header.path.resolve() not in excluded_headers
            ]

    for base_dir in base_dirs:
        for file_path in base_dir.iterdir():
            if not file_path.is_file():
                continue

            op_name = file_path.stem

            # Hand-written `src/base/` is scanned first; the generated
            # tree never overrides an already-known op.
            if op_name in ops:
                continue

            implementations = list(
                impl_headers_by_operator.get(_op_relative_type(op_name), ())
            )

            if not implementations:
                continue

            ops[op_name] = implementations

    return ops


def _generate_op_artifacts(item):
    op_name, implementations = item
    implementations = tuple(implementations)
    impl_paths = [implementation.path for implementation in implementations]
    extractor = _OperatorExtractor()
    parsed_operator = extractor(op_name)
    operator = _Operator(
        parsed_operator.name,
        parsed_operator.constructors,
        parsed_operator.calls,
        implementations,
    )
    header_name = f"{op_name}.h"
    legacy_c_source, legacy_c_header = _generate_legacy_c(operator, impl_paths)
    dispatch_declarations, dispatch_definitions = _generate_generated_dispatch_entries(
        operator
    )
    (
        call_instantiation_declarations,
        call_instantiation_definitions,
    ) = _generate_operator_call_instantiation_entries(operator)

    return {
        "op_name": op_name,
        "header_name": header_name,
        "bind_func_name": f"Bind{_op_symbol_name(op_name)}",
        "pybind11": _generate_pybind11(operator),
        "binding_source": _generate_binding_source(op_name),
        "legacy_c_source": legacy_c_source,
        "legacy_c_header": legacy_c_header,
        "dispatch_declarations": dispatch_declarations,
        "dispatch_definitions": dispatch_definitions,
        "call_instantiation_declarations": call_instantiation_declarations,
        "call_instantiation_definitions": call_instantiation_definitions,
        "impl_paths": impl_paths,
    }


def _wrapper_gen_jobs(with_torch):
    raw = os.environ.get("INFINI_OPS_WRAPPER_GEN_JOBS")

    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1

    return min(os.cpu_count() or 1, 8)


def _use_monolithic_bindings():
    value = os.environ.get("INFINI_OPS_MONOLITHIC_BINDINGS", "")

    return value.upper() in {"1", "ON", "TRUE"}


def _generate_ops_module_source(bind_func_names, op_includes=(), monolithic=False):
    bind_func_names = list(bind_func_names)
    bind_func_calls = "\n".join(
        f"{bind_func_name}(m);" for bind_func_name in bind_func_names
    )
    profile_bindings = """namespace {

pybind11::list HostRangeSummariesToPybind11(
    const std::vector<HostRangeSummary>& summaries) {
  pybind11::list result;
  for (const auto& summary : summaries) {
    pybind11::dict row;
    row["range"] = HostRangeLayerName(summary.layer);
    row["count"] = summary.count;
    row["unit"] = "ns";
    row["inclusive_mean"] = summary.inclusive_mean;
    row["inclusive_median"] = summary.inclusive_median;
    row["self_mean"] = summary.self_mean;
    row["self_median"] = summary.self_median;
    result.append(row);
  }
  return result;
}

void BindHostRangeProfileControls(pybind11::module& m) {
  m.def("_host_range_profile_compiled", &HostRangeProfiler::IsCompiled);
  m.def("_host_range_profile_start", &HostRangeProfiler::Start);
  m.def("_host_range_profile_stop", []() {
    return HostRangeSummariesToPybind11(HostRangeProfiler::Stop());
  });
  m.def("_host_range_profile_calibrate", [](std::size_t iterations) {
    return HostRangeSummariesToPybind11(
        HostRangeProfiler::Calibrate(iterations));
  }, pybind11::arg("iterations"));
}

}  // namespace
"""

    if monolithic:
        op_includes = "\n".join(op_includes)
        pre_namespace = f"""// Generated with `INFINI_OPS_MONOLITHIC_BINDINGS=1`.
{op_includes}"""
        bind_func_declarations = ""
    else:
        pre_namespace = ""
        bind_func_declarations = "\n".join(
            f"void {bind_func_name}(pybind11::module& m);"
            for bind_func_name in bind_func_names
        )

    module_calls = """TuningManager::Instance().InitializeFromEnvironment();
BindHostRangeProfileControls(m);"""
    if bind_func_calls:
        module_calls = f"{module_calls}\n{bind_func_calls}"

    return f"""#include <pybind11/pybind11.h>

#include "host_range_profiler.h"
#include "tuning.h"

{pre_namespace}

namespace infini::ops {{

{bind_func_declarations}

{profile_bindings}
PYBIND11_MODULE(ops, m) {{
{textwrap.indent(module_calls, _INDENTATION)}
}}

}}  // namespace infini::ops
"""


def _dispatch_gen_batch_size():
    raw = os.environ.get("INFINI_OPS_DISPATCH_BATCH_SIZE")

    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 8

    return 8


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An automatic wrapper generator.")

    parser.add_argument(
        "--devices",
        nargs="+",
        default="cpu",
        type=str,
        help="Devices to use. Please pick from `cpu`, `nvidia`, `cambricon`, `ascend`, `metax`, `mars`, `moore`, `iluvatar`, `hygon`, and `thead`. (default: `cpu`)",
    )

    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Include PyTorch C++ backend implementations.",
    )
    parser.add_argument(
        "--with-ninetoothed",
        action="store_true",
        help="Include NineToothed backend implementations.",
    )
    parser.add_argument(
        "--with-linked",
        action="store_true",
        help="Include linked third-party backend implementations.",
    )
    parser.add_argument(
        "--with-native-flash",
        action="store_true",
        help="Include the pinned native inference Attention provider.",
    )
    parser.add_argument(
        "--with-native-w4",
        action="store_true",
        help="Include the compiled NVIDIA Marlin providers.",
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        default=[],
        type=str,
        help="Operator allowlist to generate. Accepts names separated by spaces or commas.",
    )
    parser.add_argument(
        "--ops-config",
        type=pathlib.Path,
        help="Path to an `ops.json` operator and implementation selection.",
    )
    parser.add_argument(
        "--strict-ops",
        action="store_true",
        help="Fail if `--ops` contains operators unavailable for the active devices.",
    )

    parser.add_argument(
        "--with-triton",
        action="store_true",
        help="Include Triton backend implementations.",
    )

    args = parser.parse_args()

    for directory in (_BINDINGS_DIR, _GENERATED_SRC_DIR, _INCLUDE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    config_path = args.ops_config
    ops = _get_all_ops(
        args.devices,
        with_torch=args.with_torch,
        with_ninetoothed=args.with_ninetoothed,
        with_linked=args.with_linked,
        with_triton=args.with_triton,
        with_native_flash=args.with_native_flash,
        with_native_w4=args.with_native_w4,
    )

    if config_path is not None:
        config = ops_config.load_ops_config(config_path)
        ops = _select_ops_from_config(ops, config, config_path)

    ops = _filter_ops(
        ops,
        _normalize_op_allowlist(args.ops),
        strict=args.strict_ops,
    )

    bind_func_names = []

    jobs = _wrapper_gen_jobs(args.with_torch)

    if jobs == 1:
        artifacts = [_generate_op_artifacts(item) for item in ops.items()]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as executor:
            artifacts = list(executor.map(_generate_op_artifacts, ops.items()))

    op_names = [artifact["op_name"] for artifact in artifacts]
    expected_binding_files: set[pathlib.Path] = set()
    expected_generated_src_files: set[pathlib.Path] = set()
    expected_include_files: set[pathlib.Path] = set()
    dispatch_declarations = [
        declaration
        for artifact in artifacts
        for declaration in artifact["dispatch_declarations"]
    ]
    call_instantiation_declarations = [
        declaration
        for artifact in artifacts
        for declaration in artifact["call_instantiation_declarations"]
    ]
    use_monolithic_bindings = _use_monolithic_bindings()
    op_includes = []

    for artifact in artifacts:
        op_name = artifact["op_name"]
        source_path = _GENERATED_SRC_DIR / op_name
        header_name = artifact["header_name"]
        bind_func_name = artifact["bind_func_name"]
        binding_header_path = _BINDINGS_DIR / header_name

        _write_text_if_changed(binding_header_path, artifact["pybind11"])
        expected_binding_files.add(binding_header_path)

        if use_monolithic_bindings:
            op_includes.append(f'#include "{header_name}"')
        else:
            binding_source_path = _BINDINGS_DIR / f"{op_name}.cc"
            _write_text_if_changed(binding_source_path, artifact["binding_source"])
            expected_binding_files.add(binding_source_path)

        source_path.mkdir(exist_ok=True)
        legacy_source_path = _GENERATED_SRC_DIR / op_name / "operator.cc"
        _write_text_if_changed(legacy_source_path, artifact["legacy_c_source"])
        expected_generated_src_files.add(legacy_source_path)

        include_path = _INCLUDE_DIR / header_name
        _write_text_if_changed(include_path, artifact["legacy_c_header"])
        expected_include_files.add(include_path)

        bind_func_names.append(bind_func_name)

    dispatch_header = _generate_generated_dispatch_header(
        op_names, args.devices, dispatch_declarations
    )
    dispatch_header_path = _BINDINGS_DIR / "generated_dispatch.h"
    _write_text_if_changed(dispatch_header_path, dispatch_header)
    expected_binding_files.add(dispatch_header_path)

    call_instantiation_header = _generate_operator_call_instantiation_header(
        op_names, call_instantiation_declarations
    )
    (_INCLUDE_DIR / "infini").mkdir(exist_ok=True)
    call_instantiation_header_path = (
        _INCLUDE_DIR / "infini" / "operator_call_instantiations.h"
    )
    _write_text_if_changed(call_instantiation_header_path, call_instantiation_header)
    expected_include_files.add(call_instantiation_header_path)

    cache_clear_dispatch_source = _generate_cache_clear_dispatch_source(op_names)
    cache_clear_dispatch_source_path = _GENERATED_SRC_DIR / "cache_clear_dispatch.cc"
    _write_text_if_changed(
        cache_clear_dispatch_source_path, cache_clear_dispatch_source
    )
    expected_generated_src_files.add(cache_clear_dispatch_source_path)

    dispatch_batch_size = _dispatch_gen_batch_size()

    for dispatch_batch_index, start in enumerate(
        range(0, len(artifacts), dispatch_batch_size)
    ):
        batch = artifacts[start : start + dispatch_batch_size]
        impl_paths = list(
            dict.fromkeys(
                impl_path for artifact in batch for impl_path in artifact["impl_paths"]
            )
        )
        definitions = [
            definition
            for artifact in batch
            for definition in artifact["dispatch_definitions"]
        ]
        dispatch_source = _generate_generated_dispatch_source(impl_paths, definitions)
        dispatch_source_path = (
            _BINDINGS_DIR / f"generated_dispatch_{dispatch_batch_index}.cc"
        )
        _write_text_if_changed(dispatch_source_path, dispatch_source)
        expected_binding_files.add(dispatch_source_path)

        call_instantiation_definitions = [
            definition
            for artifact in batch
            for definition in artifact["call_instantiation_definitions"]
        ]
        call_instantiation_source = _generate_operator_call_instantiation_source(
            args.devices,
            impl_paths,
            call_instantiation_definitions,
        )
        call_instantiation_source_path = (
            _GENERATED_SRC_DIR
            / f"operator_call_instantiations_{dispatch_batch_index}.cc"
        )
        _write_text_if_changed(
            call_instantiation_source_path, call_instantiation_source
        )
        expected_generated_src_files.add(call_instantiation_source_path)

    ops_source = _generate_ops_module_source(
        bind_func_names,
        op_includes=op_includes,
        monolithic=use_monolithic_bindings,
    )

    ops_source_path = _BINDINGS_DIR / "ops.cc"
    _write_text_if_changed(ops_source_path, ops_source)
    expected_binding_files.add(ops_source_path)

    _remove_stale_files(_BINDINGS_DIR, expected_binding_files)
    _remove_stale_files(_GENERATED_SRC_DIR, expected_generated_src_files)
    _remove_stale_files(_INCLUDE_DIR, expected_include_files)
